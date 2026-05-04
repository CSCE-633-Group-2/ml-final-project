import argparse
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


DEID_DATE = re.compile(r"\[\*\*\d{4}-?\d{0,2}-?\d{0,2}\*\*\]")
DEID_HOSPITAL = re.compile(r"\[\*\*[^\]]*(?:Hospital|Location|Ward)[^\]]*\*\*\]", re.IGNORECASE)
DEID_NAME = re.compile(r"\[\*\*[^\]]*Name[^\]]*\*\*\]", re.IGNORECASE)
DEID_NUM = re.compile(r"\[\*\*[^\]]*\d+[^\]]*\*\*\]")
DEID_ANY = re.compile(r"\[\*\*[^\]]*\*\*\]")
IMG_TAG = re.compile(r"\[image[^\]]*\]", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


def set_seed_and_device(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")
	return device


def clean_text(text):
	if not isinstance(text, str):
		return ""
	text = DEID_DATE.sub("[DATE]", text)
	text = DEID_HOSPITAL.sub("[HOSPITAL]", text)
	text = DEID_NAME.sub("[NAME]", text)
	text = DEID_NUM.sub("[ID]", text)
	text = DEID_ANY.sub("[REDACTED]", text)
	text = IMG_TAG.sub("[IMG]", text)
	text = WHITESPACE.sub(" ", text)
	return text.strip()


def load_and_subsample(csv_path, n_total, seed):
	print(f"Loading {csv_path}...")
	df = pd.read_csv(csv_path, usecols=["row_id", "text", "label"]).dropna()
	df["text"] = df["text"].astype(str)
	df = df[df["text"].str.split().str.len() >= 3]
	print(f"  {len(df):,} valid rows")

	if n_total <= 0 or n_total >= len(df):
		sub = df.sample(frac=1, random_state=seed).reset_index(drop=True)
	else:
		n_pos_take = min((df.label == 1).sum(), n_total // 2)
		n_neg_take = min((df.label == 0).sum(), n_total - n_pos_take)
		pos = df[df.label == 1].sample(n=n_pos_take, random_state=seed)
		neg = df[df.label == 0].sample(n=n_neg_take, random_state=seed)
		sub = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

	pos_count = int((sub.label == 1).sum())
	neg_count = int((sub.label == 0).sum())
	print(f"  subsampled: {len(sub):,} ({pos_count:,}+/{neg_count:,}-)")
	return train_test_split(sub, test_size=0.2, random_state=seed, stratify=sub["label"])


class ClinicalDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_words, max_tokens, is_training=True):
		self.texts = list(texts)
		self.labels = list(labels) if labels is not None else None
		self.tokenizer = tokenizer
		self.max_words = max_words
		self.max_tokens = max_tokens
		self.is_training = is_training

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, i):
		text = clean_text(self.texts[i])
		if self.is_training:
			words = text.split()
			if len(words) > self.max_words:
				start = random.randint(0, len(words) - self.max_words)
				text = " ".join(words[start:start + self.max_words])

		enc = self.tokenizer(
			text,
			add_special_tokens=True,
			max_length=self.max_tokens,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			return_tensors="pt",
		)
		item = {
			"input_ids": enc["input_ids"].squeeze(0),
			"attention_mask": enc["attention_mask"].squeeze(0),
		}
		if self.labels is not None:
			item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
		return item


def make_balanced_sampler(labels):
	counts = pd.Series(labels).value_counts().to_dict()
	weights = np.array([1.0 / counts[l] for l in labels])
	return WeightedRandomSampler(
		torch.DoubleTensor(weights),
		num_samples=len(labels),
		replacement=True,
	)


def evaluate(model, loader, device, threshold=0.5):
	model.eval()
	all_probs = []
	all_y = []
	with torch.no_grad():
		for batch in loader:
			ids = batch["input_ids"].to(device)
			mask = batch["attention_mask"].to(device)
			with autocast(device_type=device.type):
				out = model(input_ids=ids, attention_mask=mask)
			probs = F.softmax(out.logits, dim=1)[:, 1].detach().cpu().numpy()
			all_probs.extend(probs.tolist())
			if "labels" in batch:
				all_y.extend(batch["labels"].cpu().numpy().tolist())

	out = {"probs": np.array(all_probs)}
	if all_y:
		y = np.array(all_y).astype(int)
		preds = (np.array(all_probs) >= threshold).astype(int)
		p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
		out.update({
			"acc": accuracy_score(y, preds),
			"precision": p,
			"recall": r,
			"f1": f,
			"auc": roc_auc_score(y, all_probs) if len(set(y)) > 1 else float("nan"),
			"preds": preds,
			"y": y,
		})
	return out


def evaluate_loss(model, loader, device, loss_fn):
	model.eval()
	all_losses = []
	with torch.no_grad():
		for batch in loader:
			ids = batch["input_ids"].to(device)
			mask = batch["attention_mask"].to(device)
			y = batch["labels"].to(device)
			with autocast(device_type=device.type):
				out = model(input_ids=ids, attention_mask=mask)
				loss = loss_fn(out.logits, y)
			all_losses.append(loss.item())

	return float(np.mean(all_losses)) if all_losses else 0.0


def train_model(args, device):
	print("\n=== TRAINING ===")
	df_train, df_val = load_and_subsample(args.train_data_path, args.n_subsample, args.seed)

	print(f"\nLoading {args.model_name}...")
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	model = AutoModelForSequenceClassification.from_pretrained(
		args.model_name,
		num_labels=2,
	).to(device)

	train_ds = ClinicalDataset(
		df_train["text"].values,
		df_train["label"].values,
		tokenizer,
		args.max_words,
		args.max_tokens,
		is_training=True,
	)
	val_ds = ClinicalDataset(
		df_val["text"].values,
		df_val["label"].values,
		tokenizer,
		args.max_words,
		args.max_tokens,
		is_training=False,
	)

	sampler = make_balanced_sampler(df_train["label"].values)
	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		sampler=sampler,
		num_workers=args.num_workers,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.lr,
		weight_decay=args.weight_decay,
	)
	total_steps = len(train_loader) * args.epochs
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=int(total_steps * args.warmup_frac),
		num_training_steps=total_steps,
	)
	scaler = GradScaler()

	best_f1 = 0.0
	train_loss = []
	val_loss = []
	for epoch in range(args.epochs):
		model.train()
		epoch_batch_loss = []
		progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [transformer]")
		for batch in progress:
			optimizer.zero_grad(set_to_none=True)
			ids = batch["input_ids"].to(device, non_blocking=True)
			mask = batch["attention_mask"].to(device, non_blocking=True)
			y = batch["labels"].to(device, non_blocking=True)

			with autocast(device_type=device.type):
				out = model(input_ids=ids, attention_mask=mask)
				loss = loss_fn(out.logits, y)

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			progress.set_postfix({"loss": f"{loss.item():.4f}"})
			epoch_batch_loss.append(loss.item())

		train_loss.append(float(np.mean(epoch_batch_loss)) if epoch_batch_loss else 0.0)
		val_loss.append(evaluate_loss(model, val_loader, device, loss_fn))
		val_metrics = evaluate(model, val_loader, device, threshold=0.5)
		print(
			"  loss: train={train:.4f}  val={val:.4f}  |  acc={acc:.4f}  f1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  AUC={auc:.4f}".format(
				train=train_loss[-1],
				val=val_loss[-1],
				**val_metrics,
			)
		)

		if val_metrics["f1"] > best_f1:
			best_f1 = val_metrics["f1"]
			os.makedirs(args.save_dir, exist_ok=True)
			model.save_pretrained(args.save_dir)
			tokenizer.save_pretrained(args.save_dir)
			print(f"  --> saved (best f1={best_f1:.4f})")

	return model, tokenizer, val_loader, train_loss, val_loss


def tune_threshold(model, val_loader, device):
	print("\n=== THRESHOLD TUNING ON VAL ===")
	out = evaluate(model, val_loader, device, threshold=0.5)
	probs, y = out["probs"], out["y"]

	best_t, best_f1 = 0.5, 0.0
	print("threshold |  acc   |  f1    |  pos_frac")
	for t in np.arange(0.30, 0.81, 0.02):
		preds = (probs >= t).astype(int)
		f1 = f1_score(y, preds, zero_division=0)
		acc = accuracy_score(y, preds)
		marker = "  <-- best" if f1 > best_f1 else ""
		print(f"   {t:.2f}    | {acc:.4f} | {f1:.4f} |  {preds.mean():.3f}{marker}")
		if f1 > best_f1:
			best_f1, best_t = f1, t

	print(f"\nBest threshold: {best_t:.2f}  (val F1 = {best_f1:.4f})")
	return float(best_t)


def make_output_path(base_path, suffix):
	base, ext = os.path.splitext(base_path)
	return f"{base}{suffix}{ext}"


def plot_loss(train_loss, validation_loss, title, save_path="./data/plots/loss_plot_transformer.png"):
	epochs = range(1, len(train_loss) + 1)
	plt.figure(figsize=(10, 5))
	plt.plot(epochs, train_loss, label="Train Loss")
	if validation_loss:
		plt.plot(epochs, validation_loss, label="Validation Loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title(title)
	plt.legend()
	plt.grid()
	plt.savefig(save_path)


def run_inference(args, device, threshold, output_suffix=""):
	print(f"\n=== INFERENCE (threshold={threshold:.2f}) ===")
	tokenizer = AutoTokenizer.from_pretrained(args.save_dir, local_files_only=True)
	model = AutoModelForSequenceClassification.from_pretrained(
		args.save_dir,
		local_files_only=True,
	).to(device)
	model.eval()

	test_files = {
		"test01": args.test1_data_path,
		"test02": args.test2_data_path,
		"test03": args.test3_data_path,
	}
	pred_files = {
		"test01": make_output_path(args.test1_output_path, output_suffix),
		"test02": make_output_path(args.test2_output_path, output_suffix),
		"test03": make_output_path(args.test3_output_path, output_suffix),
	}

	for name, test_path in test_files.items():
		if not os.path.exists(test_path):
			print(f"  skip {name}: file not found")
			continue
		df_test = pd.read_csv(test_path)
		text_col = next(
			(c for c in ["text", "TEXT", "sentence"] if c in df_test.columns),
			[c for c in df_test.columns if c.lower() != "row_id"][0],
		)
		ds = ClinicalDataset(
			df_test[text_col].values,
			None,
			tokenizer,
			args.max_words,
			args.max_tokens,
			is_training=False,
		)
		loader = DataLoader(
			ds,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=True,
		)
		probs = evaluate(model, loader, device, threshold=threshold)["probs"]
		preds = (probs >= threshold).astype(int)

		out_df = pd.DataFrame({
			"row_id": df_test["row_id"] if "row_id" in df_test.columns else np.arange(len(df_test)),
			"prediction": preds,
		})
		out_path = pred_files[name]
		out_dir = os.path.dirname(out_path)
		if out_dir:
			os.makedirs(out_dir, exist_ok=True)
		out_df.to_csv(out_path, index=False)
		print(
			f"  {name}: rows={len(out_df)}, pos_rate={preds.mean():.3f}, "
			f"dist={out_df['prediction'].value_counts().to_dict()}"
		)
		print(f"  saved to {out_path}")


def parse_args():
	parser = argparse.ArgumentParser(description="Train and evaluate a transformer text classifier.")
	parser.add_argument("--model-name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
	parser.add_argument("--max-words", type=int, default=128)
	parser.add_argument("--max-tokens", type=int, default=192)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=2)
	parser.add_argument("--lr", type=float, default=2e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-frac", type=float, default=0.05)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--n-subsample", type=int, default=200000)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--train-data-path", type=str, default="./data/mimiciii_training_data.csv")
	parser.add_argument("--test1-data-path", type=str, default="./data/test01_text_only.csv")
	parser.add_argument("--test2-data-path", type=str, default="./data/test02_text_only.csv")
	parser.add_argument("--test3-data-path", type=str, default="./data/test03_text_only.csv")
	parser.add_argument("--save-dir", type=str, default="./data/models/clinicalbert_finetuned")
	parser.add_argument("--plot-path", type=str, default="./data/plots/loss_plot_transformer.png")
	parser.add_argument("--test1-output-path", type=str, default="./data/predictions/test01-pred.csv")
	parser.add_argument("--test2-output-path", type=str, default="./data/predictions/test02-pred.csv")
	parser.add_argument("--test3-output-path", type=str, default="./data/predictions/test03-pred.csv")
	return parser.parse_args()


def main():
	args = parse_args()
	plot_dir = os.path.dirname(args.plot_path)
	if plot_dir:
		os.makedirs(plot_dir, exist_ok=True)

	print("\nArguments:\n")
	for arg in vars(args):
		print(f"{arg}: {getattr(args, arg)}")
	print("------------------------------------------------")

	device = set_seed_and_device(args.seed)
	model, tokenizer, val_loader, train_loss, val_loss = train_model(args, device)
	plot_loss(train_loss, val_loss, title="Transformer Loss Curves", save_path=args.plot_path)
	threshold = tune_threshold(model, val_loader, device)
	run_inference(args, device, threshold, output_suffix="_tuned")
	run_inference(args, device, 0.5, output_suffix="_t05")


if __name__ == "__main__":
	main()
