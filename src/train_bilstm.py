"""
BugRoute AI - Deep Learning Model (BiLSTM + Attention)
Implements PyTorch Bidirectional LSTM with word-level Attention mechanism.
Provides attention weights for explainability heatmaps.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Vocabulary:
    """Simple vocabulary builder mapping words to indices."""
    def __init__(self, max_vocab: int = 25000, unk_token: str = "<UNK>", pad_token: str = "<PAD>"):
        self.max_vocab = max_vocab
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.freqs = {}

    def fit(self, texts: List[str]):
        for text in texts:
            if not isinstance(text, str):
                continue
            for word in text.split():
                self.freqs[word] = self.freqs.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_vocab - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        tokens = text.split() if isinstance(text, str) else []
        indices = [self.word2idx.get(w, self.word2idx[self.unk_token]) for w in tokens[:max_len]]
        # Padding
        if len(indices) < max_len:
            indices += [self.word2idx[self.pad_token]] * (max_len - len(indices))
        return indices


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 128):
        self.encoded = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Attention Mechanism.
    Attention computes context vector h_attn = sum_i(alpha_i * h_i).
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=2, batch_first=True, 
            bidirectional=True, dropout=dropout
        )
        
        # Linear layer for computing attention scores
        self.attention_linear = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.bilstm(embeds)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention scoring
        attn_weights = self.attention_linear(lstm_out).squeeze(-1)  # (batch_size, seq_len)
        
        # Mask out padding tokens (index 0)
        mask = (x != 0).float()
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_weights, dim=-1)  # (batch_size, seq_len)
        
        # Weighted sum context vector
        context = torch.bmm(attn_probs.unsqueeze(1), lstm_out).squeeze(1)  # (batch_size, hidden_dim * 2)
        logits = self.fc(context)  # (batch_size, num_classes)
        
        return logits, attn_probs


def train_bilstm_model(data_dir: str = "data/processed", models_dir: str = "models", reports_dir: str = "reports", epochs: int = 5, batch_size: int = 16):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BiLSTM + Attention] Using compute device: {device}")
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    with open(os.path.join(data_dir, "class_mapping.json"), "r") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    
    # 2. Build Vocab
    vocab = Vocabulary(max_vocab=20000)
    vocab.fit(train_df['text_classical'].tolist())
    joblib.dump(vocab, os.path.join(models_dir, "bilstm_vocab.joblib"))
    
    train_dataset = TextDataset(train_df['text_classical'].tolist(), train_df['label'].tolist(), vocab)
    val_dataset = TextDataset(val_df['text_classical'].tolist(), val_df['label'].tolist(), vocab)
    test_dataset = TextDataset(test_df['text_classical'].tolist(), test_df['label'].tolist(), vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model Setup
    model = BiLSTMAttention(
        vocab_size=len(vocab.word2idx), 
        embed_dim=128, 
        hidden_dim=128, 
        num_classes=num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 4. Training Loop
    start_time = time.time()
    best_val_loss = float('inf')
    best_model_weights = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits, _ = model(batch_x)
                val_loss += criterion(logits, batch_y).item()
                
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()
            
    train_time = time.time() - start_time
    
    # Load best weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), os.path.join(models_dir, "bilstm_model.pt"))
    
    # 5. Evaluation on Test Set
    model.eval()
    all_preds, all_probas, all_targets = [], [], []
    inf_start = time.time()
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probas.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            
    inf_time_per_sample = (time.time() - inf_start) / len(test_df) * 1000  # ms per sample
    
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    _, _, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    
    # Top-3 Accuracy
    top3_preds = np.argsort(np.array(all_probas), axis=1)[:, -3:]
    top3_correct = [all_targets[i] in top3_preds[i] for i in range(len(all_targets))]
    top3_acc = float(np.mean(top3_correct))
    
    model_size_mb = os.path.getsize(os.path.join(models_dir, "bilstm_model.pt")) / (1024 * 1024)
    
    results = {
        "BiLSTM + Attention": {
            "Accuracy": round(float(acc), 4),
            "Macro-F1": round(float(f1_macro), 4),
            "Weighted-F1": round(float(f1_weighted), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "Top-3 Accuracy": round(float(top3_acc), 4),
            "Training Time (s)": round(float(train_time), 2),
            "Inference Time (ms/sample)": round(float(inf_time_per_sample), 4),
            "Model Size (MB)": round(float(model_size_mb), 2)
        }
    }
    
    print("\n--- BiLSTM + ATTENTION EVALUATION RESULTS ---")
    res_df = pd.DataFrame(results).T
    print(res_df.to_string())
    res_df.to_csv(os.path.join(reports_dir, "bilstm_results.csv"))
    return results


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_bilstm_model()
