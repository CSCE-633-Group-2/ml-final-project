import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader 
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
from transformers import AutoModel

import importlib

data_processor = importlib.import_module('lstm-data-processor')
hyperparams = importlib.import_module('lstm-hyperparams')

class MyModel(nn.Module):
    def __init__(
            self, 
            lstm_hidden_dim : int = hyperparams.LSTM_HIDDEN_DIM, 
            num_lstm_layers : int = hyperparams.NUM_LSTM_LAYERS, 
            bidirectional : bool = False,
            dropout_rate : float = 0.5,
            freeze_embeddings : bool = True
    ):
        super(MyModel, self).__init__()

        self.bidirectional = bidirectional
        
        clinical_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.embedding = clinical_bert.embeddings
        embedding_dim = clinical_bert.config.hidden_size
        
        if freeze_embeddings:
            for param in self.embedding.parameters():
                param.requires_grad = False
        
        lstm_dropout = dropout_rate if num_lstm_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=lstm_hidden_dim, 
            num_layers=num_lstm_layers, 
            bidirectional=bidirectional, 
            batch_first=True,
            dropout=lstm_dropout
        )

        fc_input_dim = (2 if bidirectional else 1) * lstm_hidden_dim

        self.attention = nn.Linear(fc_input_dim, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features=fc_input_dim, out_features=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        embeddings = self.embedding(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            position_ids=position_ids
        )
        
        out, _ = self.lstm(embeddings)

        # Calculate attention weights
        attn_weights = self.attention(out) # Shape: (batch_size, seq_len, 1)
        
        # FIX: Mask out padding tokens before Softmax so they receive 0 attention
        if attention_mask is not None:
            # attention_mask is 1 for real tokens, 0 for padding.
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)

        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * out, dim=1)

        hidden = self.dropout(context_vector)
        return self.fc(hidden).squeeze(-1)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        
        # FIX: Initialize pos_weight here once instead of every batch
        self.conservative_weight = torch.tensor([0.5]).to(self.gpu_id)

        # FIX: sync_on_compute=True ensures metrics are gathered across all GPUs 
        self.train_acc = BinaryAccuracy(sync_on_compute=True).to(self.gpu_id)
        self.train_f1 = BinaryF1Score(sync_on_compute=True).to(self.gpu_id)
        self.val_acc = BinaryAccuracy(sync_on_compute=True).to(self.gpu_id)
        self.val_f1 = BinaryF1Score(sync_on_compute=True).to(self.gpu_id)

    def _run_batch(self, source, attention_mask, targets, is_train=True) -> float:
        if is_train:
            self.optimizer.zero_grad()
        
        logits = self.model(input_ids=source, attention_mask=attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), pos_weight=self.conservative_weight)
        
        if is_train:
            loss.backward()
            self.optimizer.step()

        preds_prob = torch.sigmoid(logits)
        
        if is_train:
            self.train_acc.update(preds_prob, targets)
            self.train_f1.update(preds_prob, targets)
        else:
            self.val_acc.update(preds_prob, targets)
            self.val_f1.update(preds_prob, targets)

        return loss.item()

    def _run_epoch(self, epoch) -> tuple[float, float, float]:
        self.train_data.sampler.set_epoch(epoch) # type: ignore

        self.train_acc.reset()
        self.train_f1.reset()
        total_loss = 0.0 
        
        for i, (source, attention_mask, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            attention_mask = attention_mask.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            total_loss += self._run_batch(source, attention_mask, targets, is_train=True)

            if self.gpu_id == 0 and i % 100 == 0:
                print(f"Epoch {epoch + 1} | Batch {i}/{len(self.train_data)} | Current Avg Loss: {total_loss / (i + 1):.4f}")

        avg_loss = total_loss / len(self.train_data)
        
        loss_tensor = torch.tensor(avg_loss).to(self.gpu_id)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)

        return loss_tensor.item(), self.train_acc.compute().item(), self.train_f1.compute().item()

    def _save_checkpoint(self, epoch : int):
        assert isinstance(self.model.module, nn.Module), "Cannot save non-module object"

        ckp = self.model.module.state_dict()
        PATH = "./data/models/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch + 1} | Training checkpoint saved at {PATH}")

    def _save_best_model(self, epoch : int, metric_name : str, best_metric : float):
        assert isinstance(self.model.module, nn.Module), "Cannot save non-module object"

        state = self.model.module.state_dict()
        PATH = "./data/models/best_model.pt"
        torch.save(state, PATH)
        print(f"Epoch {epoch + 1} | Got new best {metric_name} of {best_metric} | Model save at {PATH}")

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float, float]:
        self.val_acc.reset()
        self.val_f1.reset()
        total_loss = 0.0

        assert isinstance(self.model.module, nn.Module), "Must be a module"
        assert self.val_data is not None, "Validation data cannot be empty during evaluation."

        self.model.eval()
        
        for features, attention_mask, targets in self.val_data:
            features = features.to(self.gpu_id)
            attention_mask = attention_mask.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            total_loss += self._run_batch(features, attention_mask, targets, is_train=False)

        self.model.train()
        
        avg_loss = total_loss / len(self.val_data)
        loss_tensor = torch.tensor(avg_loss).to(self.gpu_id)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
        
        return loss_tensor.item(), self.val_acc.compute().item(), self.val_f1.compute().item()

    def train(self, max_epochs: int):
        best_acc = 0.0

        history = {
            "train_loss": [], "train_acc": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_f1": []
        }

        for epoch in range(max_epochs):
            t_loss, t_acc, t_f1 = self._run_epoch(epoch)

            v_loss, v_acc, v_f1 = 0.0, 0.0, 0.0
            
            if self.val_data is not None:
                v_loss, v_acc, v_f1 = self._evaluate()
                self.scheduler.step(v_loss)

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["train_f1"].append(t_f1)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
            history["val_f1"].append(v_f1)

            if self.gpu_id == 0:
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch + 1}/{max_epochs} | LR: {current_lr:.2e} | "
                      f"Train Loss: {t_loss:.4f} - Acc: {t_acc:.4f} - F1: {t_f1:.4f} | "
                      f"Val Loss: {v_loss:.4f} - Acc: {v_acc:.4f} - F1: {v_f1:.4f}")


                if self.val_data is not None and v_acc > best_acc:
                    best_acc = v_acc
                    self._save_best_model(epoch, "Accuracy", best_acc)

                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)

            torch.distributed.barrier(device_ids=[self.gpu_id])

        return history