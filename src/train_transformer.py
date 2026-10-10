"""
BugRoute AI - Ultra-Fast Transformer Fine-Tuning Module
Fine-tunes HuggingFace Transformers:
1. DistilBERT (distilbert-base-uncased) - General NLP Transformer
2. CodeBERT (microsoft/codebert-base) - Code & Natural Language Transformer

Optimized for fast CPU/GPU execution with frozen embeddings & top-layer tuning.
"""

import os
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, logging as hf_logging
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Suppress non-critical HuggingFace warning messages
hf_logging.set_verbosity_error()


class TransformerBugDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int = 64):
        self.texts = [str(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_transformer(
    model_name: str, 
    model_short_name: str,
    data_dir: str = "data/processed", 
    models_dir: str = "models", 
    reports_dir: str = "reports",
    epochs: int = 3, 
    batch_size: int = 32, 
    lr: float = 2e-4,
    max_len: int = 64,
    freeze_backbone: bool = True
) -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================================")
    print(f" Fine-Tuning Transformer: {model_name} ({model_short_name}) on {device}")
    print(f" Mode: {'Fast CPU (Frozen Backbone)' if (freeze_backbone and device.type=='cpu') else 'Full Fine-Tuning'}")
    print(f" Sequence Length: {max_len} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"==================================================================")
    
    # 1. Load Datasets
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    with open(os.path.join(data_dir, "class_mapping.json"), "r") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    
    # 2. Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = TransformerBugDataset(train_df['text_transformer'].tolist(), train_df['label'].tolist(), tokenizer, max_len)
    val_dataset = TransformerBugDataset(val_df['text_transformer'].tolist(), val_df['label'].tolist(), tokenizer, max_len)
    test_dataset = TransformerBugDataset(test_df['text_transformer'].tolist(), test_df['label'].tolist(), tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model Initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    ).to(device)
    
    # On CPU, freeze backbone layers to make forward/backward pass 10x faster!
    if freeze_backbone and device.type == "cpu":
        base_model = getattr(model, "distilbert", getattr(model, "roberta", None))
        if base_model:
            for param in base_model.parameters():
                param.requires_grad = False
            print("[Info] Base transformer layers frozen for ultra-fast CPU training.")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    
    # 4. Fine-Tuning Loop
    start_time = time.time()
    best_val_acc = 0.0
    save_path = os.path.join(models_dir, model_short_name)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        # Validation evaluation after each epoch
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
    train_time = time.time() - start_time
    
    # 5. Final Evaluation on Holdout Test Set
    model.eval()
    all_preds, all_probas, all_targets = [], [], []
    inf_start = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probas.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(batch['labels'].numpy())
            
    inf_time_per_sample = (time.time() - inf_start) / len(test_df) * 1000  # ms per sample
    
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    _, _, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    
    # Top-3 Accuracy
    top3_preds = np.argsort(np.array(all_probas), axis=1)[:, -3:]
    top3_correct = [all_targets[i] in top3_preds[i] for i in range(len(all_targets))]
    top3_acc = float(np.mean(top3_correct))
    
    # Calculate model folder size
    model_size_mb = sum(
        os.path.getsize(os.path.join(save_path, f)) 
        for f in os.listdir(save_path) 
        if os.path.isfile(os.path.join(save_path, f))
    ) / (1024 * 1024)
    
    results = {
        model_short_name: {
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
    
    print(f"\n--- {model_short_name} TEST SET FINAL BENCHMARKS ---")
    res_df = pd.DataFrame(results).T
    print(res_df.to_string())
    res_df.to_csv(os.path.join(reports_dir, f"{model_short_name.lower()}_results.csv"))
    return results


def train_all_transformers():
    # 1. DistilBERT
    res_distilbert = train_transformer(
        model_name="distilbert-base-uncased",
        model_short_name="DistilBERT",
        epochs=3,
        batch_size=32,
        lr=3e-4,
        max_len=64,
        freeze_backbone=True
    )
    
    # 2. CodeBERT
    res_codebert = train_transformer(
        model_name="microsoft/codebert-base",
        model_short_name="CodeBERT",
        epochs=3,
        batch_size=32,
        lr=3e-4,
        max_len=64,
        freeze_backbone=True
    )
    
    return {**res_distilbert, **res_codebert}

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_all_transformers()
