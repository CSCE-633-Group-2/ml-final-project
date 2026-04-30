from data_processor import load_and_preprocess_data, Vocabulary
import numpy as np
import torch
import torch.nn as nn
import pandas as pd 
import math
from tqdm import tqdm
import string
import re
import argparse
import os
import torchtext.vocab as tv

import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, vocab_size, num_layers, embedding_dim=100, hidden_dim=128, output_dim=1, embedding_matrix=None):
        super(RNN, self).__init__()
        if embedding_matrix is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, padding_idx=0, freeze=False
            )
        self.rnn = nn.RNN(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=0.3 if num_layers > 1 else 0.0,
                nonlinearity='tanh', 
            bidirectional=True, 
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)
        
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        
        # Mask padding tokens
        mask = (x != 0).unsqueeze(-1).float()
        masked = output * mask
        
        mean_pool = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        max_pool, _ = masked.masked_fill(mask == 0, -1e9).max(dim=1)
        
        combined = torch.cat([mean_pool, max_pool], dim=-1)
        return self.fc(self.dropout(combined))

def load_glove_embeddings(vocab, embedding_dim=100, device='cpu'):
    glove = tv.GloVe(name='6B', dim=embedding_dim)
    embedding_matrix = torch.zeros(vocab.size, embedding_dim, dtype=torch.float32)
    hits = 0
    for word, idx in vocab.word2idx.items():
        if word in glove.stoi:
            vec = glove[word]
            if not isinstance(vec, torch.Tensor):
                vec = torch.tensor(vec)
            embedding_matrix[idx] = vec.to(dtype=torch.float32)
            hits += 1
    print(f"GloVe hits: {hits}/{vocab.size}")
    return embedding_matrix

def compute_pos_weight(iterator):
    positive_count = 0
    negative_count = 0
    for batch in iterator:
        labels = batch[1].view(-1)
        positive_count += (labels == 1).sum().item()
        negative_count += (labels == 0).sum().item()

    positive_count = max(positive_count, 1)
    return torch.tensor([negative_count / positive_count], dtype=torch.float32)

def collect_predictions(model, iterator, device):
    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for indices, labels in iterator:
            indices = indices.to(device)
            labels = labels.to(device).squeeze(1)
            lengths = (indices != 0).sum(dim=1)
            logits = model(indices, lengths).squeeze(1)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_logits:
        return torch.empty(0), torch.empty(0)

    return torch.cat(all_logits), torch.cat(all_labels)

def find_best_threshold(model, iterator, device):
    logits, labels = collect_predictions(model, iterator, device)
    if logits.numel() == 0:
        return 0.5, 0.0, 0.0

    probabilities = torch.sigmoid(logits)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        predicted_labels = (probabilities >= threshold).long()
        true_positive = ((predicted_labels == 1) & (labels == 1)).sum().item()
        false_positive = ((predicted_labels == 1) & (labels == 0)).sum().item()
        false_negative = ((predicted_labels == 0) & (labels == 1)).sum().item()
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_beta2 = (1 + 4) * precision * recall / (4 * precision + recall + 1e-8)

        if f1_beta2 > best_score:
            best_score = f1_beta2
            best_threshold = float(threshold)

    return best_threshold, best_score

def train(model, iterator, optimizer, criterion, device, val_loader, num_epochs=5, save_model_path="./data/models/rnn.pt", grad_clip=1.0):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        epoch_batch_loss = []
        epoch_correct = 0
        epoch_total = 0

        for indices, labels in tqdm(iterator, desc=f"Epoch {epoch + 1}/{num_epochs} [rnn]"):
            indices, labels = indices.to(device), labels.to(device).squeeze(1)
            lengths = (indices != 0).sum(dim=1)
            optimizer.zero_grad()
            predictions = model(indices, lengths).squeeze(1)
            loss = criterion(predictions, labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_batch_loss.append(loss.item())

            predicted_labels = (torch.sigmoid(predictions) >= 0.5).long()
            epoch_correct += (predicted_labels == labels.long()).sum().item()
            epoch_total += labels.size(0)

        train_loss.append(float(np.mean(epoch_batch_loss)))
        train_acc.append(float(epoch_correct / epoch_total) if epoch_total > 0 else 0.0)

        eval_loss, eval_acc = evaluate(model, val_loader, criterion, device, return_accuracy=True)
        val_loss.append(eval_loss)
        val_acc.append(eval_acc)
        if eval_acc > best_val_acc:
            best_val_acc = eval_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss[-1]:.4f} - Train Acc: {train_acc[-1]:.4f} - Val Loss: {val_loss[-1]:.4f} - Val Acc: {val_acc[-1]:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(model.state_dict(), save_model_path)


    return train_loss, train_acc, val_loss, val_acc

def evaluate(model, iterator, criterion, device, return_accuracy=False, save_predictions=False, output_path="./data/predictions/test01-pred.csv", threshold=0.5):
    test_loss = []
    correct = 0
    total = 0
    
    if  save_predictions:
        predictions_list = []
        model.eval()
        with torch.no_grad():
            for indices, labels in tqdm(iterator):
                indices = indices.to(device)
                lengths = (indices != 0).sum(dim=1)
                predictions = model(indices, lengths).squeeze(1)
                predicted_labels = (torch.sigmoid(predictions) >= threshold).long()
                predictions_list.extend(predicted_labels.cpu().numpy())
        pred_df = pd.DataFrame({'row_id': range(len(predictions_list)), 'prediction': predictions_list})
        pred_df.to_csv(output_path, index=False)
    else:
        model.eval()
        with torch.no_grad():
            for indices, labels in tqdm(iterator):
                indices, labels = indices.to(device), labels.to(device).squeeze(1)
                lengths = (indices != 0).sum(dim=1)
                predictions = model(indices, lengths).squeeze(1)
                loss = criterion(predictions, labels.float())
                test_loss.append(loss.item())

                predicted_labels = (torch.sigmoid(predictions) >= threshold).long()
                correct += (predicted_labels == labels.long()).sum().item()
                total += labels.size(0)
        
        mean_loss = float(np.mean(test_loss))
        accuracy = float(correct / total) if total > 0 else 0.0

        if return_accuracy:
            return mean_loss, accuracy
        return mean_loss

def plot_loss(train_loss, validation_loss, title, save_path="./data/plots/loss_plot_rnn.png"):
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an RNN text classifier.")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-data-path", type=str, default="./data/mimiciii_training_data.csv")
    parser.add_argument("--test1-data-path", type=str, default="./data/test01_text_only.csv")
    parser.add_argument("--test2-data-path", type=str, default="./data/test02_text_only.csv")
    parser.add_argument("--test3-data-path", type=str, default="./data/test03_text_only.csv")
    parser.add_argument("--test1-label-path", type=str, default="./data/test01-pred.csv")
    parser.add_argument("--model-path", type=str, default="./data/models/rnn.pt")
    parser.add_argument("--plot-path", type=str, default="./data/plots/loss_plot_rnn.png")
    parser.add_argument("--test1-output-path", type=str, default="./data/predictions/test01-pred.csv")
    parser.add_argument("--test2-output-path", type=str, default="./data/predictions/test02-pred.csv")
    parser.add_argument("--test3-output-path", type=str, default="./data/predictions/test03-pred.csv")
    return parser.parse_args()

def main():
    args = parse_args()

    for path in (
        args.model_path,
        args.plot_path,
        args.test1_output_path,
        args.test2_output_path,
        args.test3_output_path,
    ):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    train_loader, val_loader, vocab = load_and_preprocess_data(
        args.train_data_path,
        data_type='train_val',
        model_type='rnn',
        max_vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    test_loader = load_and_preprocess_data(
        args.test1_data_path,
        data_type='test',
        shared_vocab=vocab,
        model_type='rnn',
        batch_size=args.batch_size,
        max_len=args.max_len,
        label_path=args.test1_label_path,
    )

    # Print arguments for verification
    print("\nArguments:\n")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")   

    print("------------------------------------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    embedding_matrix = load_glove_embeddings(vocab, embedding_dim=args.embedding_dim)
    # Ensure embedding matrix is float32 and on the same device as the model
    if isinstance(embedding_matrix, torch.Tensor):
        embedding_matrix = embedding_matrix.to(dtype=torch.float32, device=device)
    model = RNN(vocab_size=vocab.size, num_layers=args.num_layers, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, output_dim=1, embedding_matrix=embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # pos_weight = compute_pos_weight(train_loader).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    # print(f"Using pos_weight={pos_weight.item():.4f}")

    train_loss, train_acc, val_loss, val_acc = train(model, train_loader, optimizer, criterion, device, val_loader, num_epochs=args.epochs, save_model_path=args.model_path, grad_clip=1.0)
    best_threshold, best_val_score = find_best_threshold(model, val_loader, device)
    # test_loss, test_acc = evaluate(model, test_loader, criterion, device, return_accuracy=True, threshold=best_threshold)
    plot_loss(train_loss, val_loss, title="RNN Loss Curves", save_path=args.plot_path)

    print(f"Best Validation Accuracy (0.5 threshold): {max(val_acc):.4f}")
    print(f"Best Validation Threshold: {best_threshold:.2f}")
    print(f"Best Validation Score (tuned threshold): {best_val_score:.4f}")
    # print(f"Best Validation F1 (tuned threshold): {best_val_f1:.4f}")
    # print(f"Test Accuracy: {test_acc:.4f}")

    test_1_loader = load_and_preprocess_data(args.test1_data_path, data_type='test', shared_vocab=vocab, model_type='rnn', batch_size=args.batch_size, max_len=args.max_len)
    evaluate(model, test_1_loader, criterion, device, save_predictions=True, output_path=args.test1_output_path, threshold=best_threshold)

    test_2_loader = load_and_preprocess_data(args.test2_data_path, data_type='test', shared_vocab=vocab, model_type='rnn', batch_size=args.batch_size, max_len=args.max_len)
    evaluate(model, test_2_loader, criterion, device, save_predictions=True, output_path=args.test2_output_path, threshold=best_threshold)

    test_3_loader = load_and_preprocess_data(args.test3_data_path, data_type='test', shared_vocab=vocab, model_type='rnn', batch_size=args.batch_size, max_len=args.max_len)
    evaluate(model, test_3_loader, criterion, device, save_predictions=True, output_path=args.test3_output_path, threshold=best_threshold)

    torch.save(model.state_dict(), args.model_path)

if __name__ == "__main__":
    main()