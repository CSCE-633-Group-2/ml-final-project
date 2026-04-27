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

import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)

        if lengths is None:
            lengths = (x != 0).sum(dim=1)
        lengths = lengths.clamp(min=1)
        batch_indices = torch.arange(output.size(0), device=output.device)
        last_valid_timestep = lengths - 1
        last_output = output[batch_indices, last_valid_timestep]

        last_output = self.dropout(last_output)
        out = self.fc(last_output)
        return out

def train(model, iterator, optimizer, criterion, device, val_loader, num_epochs=5, save_model_path="./data/models/rnn.pt"):
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

def evaluate(model, iterator, criterion, device, return_accuracy=False, save_predictions=False, output_path="./data/predictions/test01-pred.csv"):
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
                predicted_labels = (torch.sigmoid(predictions) >= 0.5).long()
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

                predicted_labels = (torch.sigmoid(predictions) >= 0.5).long()
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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-data-path", type=str, default="./data/mimiciii_training_data.csv")
    parser.add_argument("--test1-data-path", type=str, default="./data/test01_text_only.csv")
    parser.add_argument("--test2-data-path", type=str, default="./data/test02_text_only.csv")
    parser.add_argument("--test3-data-path", type=str, default="./data/test03_text_only.csv")
    parser.add_argument("--test1-label-path", type=str, default="./data/test01-pred(example).csv")
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
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")   

    print("------------------------------------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = RNN(vocab_size=vocab.size, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, train_acc, val_loss, val_acc = train(model, train_loader, optimizer, criterion, device, val_loader, num_epochs=args.epochs, save_model_path=args.model_path)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, return_accuracy=True)
    plot_loss(train_loss, val_loss, title="RNN Loss Curves", save_path=args.plot_path)

    print(f"Best Validation Accuracy: {max(val_acc):.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    test_1_loader = load_and_preprocess_data(args.test1_data_path, data_type='test', shared_vocab=vocab, model_type='rnn', batch_size=args.batch_size, max_len=args.max_len)
    evaluate(model, test_1_loader, criterion, device, save_predictions=True, output_path=args.test1_output_path)

    test_2_loader = load_and_preprocess_data(args.test2_data_path, data_type='test', shared_vocab=vocab, model_type='rnn', batch_size=args.batch_size, max_len=args.max_len)
    evaluate(model, test_2_loader, criterion, device, save_predictions=True, output_path=args.test2_output_path)

    test_3_loader = load_and_preprocess_data(args.test3_data_path, data_type='test', shared_vocab=vocab, model_type='rnn', batch_size=args.batch_size, max_len=args.max_len)
    evaluate(model, test_3_loader, criterion, device, save_predictions=True, output_path=args.test3_output_path)

    torch.save(model.state_dict(), args.model_path)

if __name__ == "__main__":
    main()