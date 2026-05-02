#if needed to run in google collab
# from google.colab import drive
# drive.mount('/content/drive')

import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

class Config:
    """Static configuration for data paths, model setup, and training settings."""

    PROJECT_DIR     = "/content/drive/MyDrive/ML-PROJECT"
    TRAIN_DATA_PATH = f"{PROJECT_DIR}/mimiciii_training_data.csv"
    TEST_FILES = {
        "test01": f"{PROJECT_DIR}/test01_text_only.csv",
        "test02": f"{PROJECT_DIR}/test02_text_only.csv",
        "test03": f"{PROJECT_DIR}/test03_text_only.csv",
    }
    SAVE_DIR  = f"{PROJECT_DIR}/clinicalbert_finetuned"

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

    MAX_WORDS  = 128
    MAX_TOKENS = 192

    BATCH_SIZE = 32       
    EPOCHS     = 2
    LR         = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_FRAC  = 0.05
    GRAD_CLIP    = 1.0
    SEED         = 42

    N_SUBSAMPLE  = 200_000

    DEFAULT_THRESHOLD = 0.5
    
PRED_FILES = {k: f"{Config.PROJECT_DIR}/{k}-pred.csv" for k in Config.TEST_FILES}

def set_seed_and_device(seed):
    """Seed random generators and select the active compute device.

    Args:
        seed: Integer seed used for Python, NumPy, and PyTorch RNGs.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

DEID_DATE     = re.compile(r'\[\*\*\d{4}-?\d{0,2}-?\d{0,2}\*\*\]')
DEID_HOSPITAL = re.compile(r'\[\*\*[^\]]*(?:Hospital|Location|Ward)[^\]]*\*\*\]', re.IGNORECASE)
DEID_NAME     = re.compile(r'\[\*\*[^\]]*Name[^\]]*\*\*\]', re.IGNORECASE)
DEID_NUM      = re.compile(r'\[\*\*[^\]]*\d+[^\]]*\*\*\]')
DEID_ANY      = re.compile(r'\[\*\*[^\]]*\*\*\]')
IMG_TAG       = re.compile(r'\[image[^\]]*\]', re.IGNORECASE)
WHITESPACE    = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """Normalize de-identification placeholders while preserving clinical signal.

    Args:
        text: Raw clinical text fragment.

    Returns:
        str: Cleaned text with placeholder normalization and compact whitespace.
    """
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
    """Load training data and return stratified train/validation splits.

    Args:
        csv_path: Path to input CSV containing row_id, text, and label columns.
        n_total: Target number of rows after balanced subsampling.
        seed: Random seed for deterministic sampling.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and validation dataframes.
    """
    print(f"Loading {csv_path}…")
    df = pd.read_csv(csv_path, usecols=["row_id", "text", "label"]).dropna()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.split().str.len() >= 3]
    print(f"  {len(df):,} valid rows")

    n_pos_take = min((df.label == 1).sum(), n_total // 2)
    n_neg_take = min((df.label == 0).sum(), n_total - n_pos_take)
    pos = df[df.label == 1].sample(n=n_pos_take, random_state=seed)
    neg = df[df.label == 0].sample(n=n_neg_take, random_state=seed)
    sub = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  subsampled: {len(sub):,} ({(sub.label==1).sum():,}+/{(sub.label==0).sum():,}-)")

    return train_test_split(sub, test_size=0.05, random_state=seed, stratify=sub["label"])

class ClinicalDataset(Dataset):
    """PyTorch dataset for clinical text tokenization and optional label handling.

    Long notes are randomly cropped only during training to increase text coverage
    across epochs while keeping sequence length bounded.
    """

    def __init__(self, texts, labels, tokenizer, is_training=True):
        """Initialize the dataset.

        Args:
            texts: Iterable of input note fragments.
            labels: Iterable of binary labels, or None for inference datasets.
            tokenizer: Hugging Face tokenizer instance.
            is_training: Whether to apply random word-window cropping.
        """
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.is_training = is_training

    def __len__(self):
        """Return dataset size.

        Returns:
            int: Number of examples.
        """
        return len(self.texts)

    def __getitem__(self, i):
        """Fetch and tokenize one example.

        Args:
            i: Example index.

        Returns:
            dict[str, torch.Tensor]: Tokenized inputs and optional label tensor.
        """
        text = clean_text(self.texts[i])

        if self.is_training:
            words = text.split()
            if len(words) > Config.MAX_WORDS:
                start = random.randint(0, len(words) - Config.MAX_WORDS)
                text = " ".join(words[start:start + Config.MAX_WORDS])

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_TOKENS,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item


def make_balanced_sampler(labels):
    """Build a weighted sampler to balance classes during mini-batch sampling.

    Args:
        labels: Iterable of binary class labels.

    Returns:
        WeightedRandomSampler: Sampler with inverse-frequency class weights.
    """
    counts = pd.Series(labels).value_counts().to_dict()
    weights = np.array([1.0 / counts[l] for l in labels])
    return WeightedRandomSampler(
        torch.DoubleTensor(weights),
        num_samples=len(labels),
        replacement=True,
    )

def train_model(device):
    """Train the model and persist the best checkpoint by validation F1.

    Args:
        device: Torch device used for forward and backward passes.

    Returns:
        tuple: Trained model, tokenizer, and validation dataloader.
    """
    print("\n=== TRAINING ===")
    df_train, df_val = load_and_subsample(Config.TRAIN_DATA_PATH, Config.N_SUBSAMPLE, Config.SEED)

    print(f"\nLoading {Config.MODEL_NAME}…")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2
    ).to(device)

    train_ds = ClinicalDataset(df_train["text"].values, df_train["label"].values, tokenizer, is_training=True)
    val_ds   = ClinicalDataset(df_val["text"].values, df_val["label"].values, tokenizer, is_training=False)

    sampler = make_balanced_sampler(df_train["label"].values)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR,
                                   weight_decay=Config.WEIGHT_DECAY)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * Config.WARMUP_FRAC),
        num_training_steps=total_steps,
    )
    scaler = GradScaler()

    best_f1 = 0.0
    for epoch in range(Config.EPOCHS):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [train]")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            ids   = batch["input_ids"].to(device, non_blocking=True)
            mask  = batch["attention_mask"].to(device, non_blocking=True)
            y     = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type=device.type):
                out = model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(out.logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        val_metrics = evaluate(model, val_loader, device)
        print(f"  val: acc={val_metrics['acc']:.4f}  f1={val_metrics['f1']:.4f}  "
              f"P={val_metrics['precision']:.4f}  R={val_metrics['recall']:.4f}  "
              f"AUC={val_metrics['auc']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            os.makedirs(Config.SAVE_DIR, exist_ok=True)
            model.save_pretrained(Config.SAVE_DIR)
            tokenizer.save_pretrained(Config.SAVE_DIR)
            print(f"  --> saved (best f1={best_f1:.4f})")

    return model, tokenizer, val_loader

def evaluate(model, loader, device, return_probs=False):
    """Run model inference on a dataloader and optionally compute metrics.

    Args:
        model: Sequence classification model.
        loader: DataLoader producing tokenized batches.
        device: Torch device for inference.
        return_probs: Reserved for compatibility.

    Returns:
        dict: Contains probability scores and, when labels exist, metrics and labels.
    """
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with autocast(device_type=device.type):
                out = model(input_ids=ids, attention_mask=mask)
            probs = F.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            if "labels" in batch:
                all_y.extend(batch["labels"].cpu().numpy().tolist())

    out = {"probs": np.array(all_probs)}
    if all_y:
        y = np.array(all_y).astype(int)
        preds = (np.array(all_probs) >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        out.update({
            "acc": accuracy_score(y, preds),
            "precision": p, "recall": r, "f1": f,
            "auc": roc_auc_score(y, all_probs) if len(set(y)) > 1 else float("nan"),
            "preds": preds, "y": y,
        })
    return out


def tune_threshold(model, val_loader, device):
    """Select the best probability threshold using validation F1.

    Args:
        model: Trained sequence classification model.
        val_loader: Validation dataloader with labels.
        device: Torch device for inference.

    Returns:
        float: Threshold that maximizes F1 on validation data.
    """
    print("\n=== THRESHOLD TUNING ON VAL ===")
    out = evaluate(model, val_loader, device)
    probs, y = out["probs"], out["y"]

    best_t, best_f1 = 0.5, 0.0
    print("threshold |  acc   |  f1    |  pos_frac")
    for t in np.arange(0.30, 0.81, 0.02):
        preds = (probs >= t).astype(int)
        f1  = f1_score(y, preds, zero_division=0)
        acc = accuracy_score(y, preds)
        marker = "  <-- best" if f1 > best_f1 else ""
        print(f"   {t:.2f}    | {acc:.4f} | {f1:.4f} |  {preds.mean():.3f}{marker}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nBest threshold: {best_t:.2f}  (val F1 = {best_f1:.4f})")
    return float(best_t)

def run_inference(device, threshold):
    """Generate predictions for each configured test file and save CSV outputs.

    Args:
        device: Torch device for model inference.
        threshold: Probability cutoff used for binary predictions.

    Returns:
        None
    """
    print(f"\n=== INFERENCE (threshold={threshold:.2f}) ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.SAVE_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.SAVE_DIR, local_files_only=True
    ).to(device)
    model.eval()

    for name, test_path in Config.TEST_FILES.items():
        if not os.path.exists(test_path):
            print(f"  skip {name}: file not found")
            continue
        df_test = pd.read_csv(test_path)
        text_col = next((c for c in ["text", "TEXT", "sentence"] if c in df_test.columns),
                        [c for c in df_test.columns if c.lower() != "row_id"][0])

        ds = ClinicalDataset(df_test[text_col].values, None, tokenizer, is_training=False)
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
        probs = evaluate(model, loader, device)["probs"]
        preds = (probs >= threshold).astype(int)

        out_df = pd.DataFrame({
            "row_id":     df_test["row_id"] if "row_id" in df_test.columns else np.arange(len(df_test)),
            "prediction": preds,
        })
        out_df.to_csv(Config.PRED_FILES[name], index=False)
        print(f"  {name}: rows={len(out_df)}, pos_rate={preds.mean():.3f}, "
              f"dist={out_df['prediction'].value_counts().to_dict()}")
        print(f"  saved to {Config.PRED_FILES[name]}")

if __name__ == "__main__":
    device = set_seed_and_device(Config.SEED)
    model, tokenizer, val_loader = train_model(device)
    threshold = tune_threshold(model, val_loader, device)
    run_inference(device, threshold)