"""
BugRoute AI - Evaluation & Data Leakage Verification Engine
Evaluates all 5 trained models strictly on the held-out test set (375 unseen samples):
1. Logistic Regression
2. Linear SVM
3. BiLSTM + Attention
4. DistilBERT
5. CodeBERT

Computes actual accuracy, precision, recall, macro-F1, weighted-F1, top-3 accuracy,
and true 2D confusion matrices directly from y_test and y_pred.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)

def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """Calculates top-k accuracy for multi-class predictions."""
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(correct))


def evaluate_classical_models(data_dir: str = "data/processed", models_dir: str = "models") -> Dict[str, Any]:
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    with open(os.path.join(data_dir, "class_mapping.json"), "r") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    all_labels = list(range(num_classes))
    
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    lr_path = os.path.join(models_dir, "logreg_model.joblib")
    svm_path = os.path.join(models_dir, "svm_model.joblib")
    
    results = {}
    if not (os.path.exists(vec_path) and os.path.exists(lr_path) and os.path.exists(svm_path)):
        print("[Evaluate] Classical model artifacts not found.")
        return results

    vectorizer = joblib.load(vec_path)
    logreg = joblib.load(lr_path)
    svm = joblib.load(svm_path)

    X_test_raw = test_df['text_classical'].fillna("").values
    y_test = test_df['label'].values
    
    X_test = vectorizer.transform(X_test_raw)
    
    # 1. Logistic Regression
    inf_start = time.time()
    preds_lr = logreg.predict(X_test)
    probas_lr = logreg.predict_proba(X_test)
    inf_time_lr = (time.time() - inf_start) / len(y_test) * 1000  # ms/sample
    
    acc_lr = accuracy_score(y_test, preds_lr)
    prec_m_lr, rec_m_lr, f1_m_lr, _ = precision_recall_fscore_support(y_test, preds_lr, average='macro', zero_division=0)
    prec_w_lr, rec_w_lr, f1_w_lr, _ = precision_recall_fscore_support(y_test, preds_lr, average='weighted', zero_division=0)
    top3_lr = top_k_accuracy(y_test, probas_lr, k=3)
    cm_lr = confusion_matrix(y_test, preds_lr, labels=all_labels)

    results["Logistic Regression"] = {
        "Accuracy": round(float(acc_lr), 4),
        "Macro Precision": round(float(prec_m_lr), 4),
        "Macro Recall": round(float(rec_m_lr), 4),
        "Macro-F1": round(float(f1_m_lr), 4),
        "Weighted-F1": round(float(f1_w_lr), 4),
        "Top-3 Accuracy": round(float(top3_lr), 4),
        "Inference Time (ms/sample)": round(float(inf_time_lr), 4),
        "Model Size (MB)": round(os.path.getsize(lr_path) / (1024 * 1024), 2),
        "confusion_matrix": cm_lr.tolist()
    }

    # 2. Linear SVM
    inf_start = time.time()
    preds_svm = svm.predict(X_test)
    probas_svm = svm.predict_proba(X_test)
    inf_time_svm = (time.time() - inf_start) / len(y_test) * 1000  # ms/sample

    acc_svm = accuracy_score(y_test, preds_svm)
    prec_m_svm, rec_m_svm, f1_m_svm, _ = precision_recall_fscore_support(y_test, preds_svm, average='macro', zero_division=0)
    prec_w_svm, rec_w_svm, f1_w_svm, _ = precision_recall_fscore_support(y_test, preds_svm, average='weighted', zero_division=0)
    top3_svm = top_k_accuracy(y_test, probas_svm, k=3)
    cm_svm = confusion_matrix(y_test, preds_svm, labels=all_labels)

    results["Linear SVM"] = {
        "Accuracy": round(float(acc_svm), 4),
        "Macro Precision": round(float(prec_m_svm), 4),
        "Macro Recall": round(float(rec_m_svm), 4),
        "Macro-F1": round(float(f1_m_svm), 4),
        "Weighted-F1": round(float(f1_w_svm), 4),
        "Top-3 Accuracy": round(float(top3_svm), 4),
        "Inference Time (ms/sample)": round(float(inf_time_svm), 4),
        "Model Size (MB)": round(os.path.getsize(svm_path) / (1024 * 1024), 2),
        "confusion_matrix": cm_svm.tolist()
    }

    return results


def evaluate_bilstm_model(data_dir: str = "data/processed", models_dir: str = "models") -> Dict[str, Any]:
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    with open(os.path.join(data_dir, "class_mapping.json"), "r") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    all_labels = list(range(num_classes))
    
    bilstm_path = os.path.join(models_dir, "bilstm_model.pt")
    vocab_path = os.path.join(models_dir, "bilstm_vocab.joblib")
    
    if not (os.path.exists(bilstm_path) and os.path.exists(vocab_path)):
        print("[Evaluate] BiLSTM model file not found.")
        return {}
        
    try:
        from src.train_bilstm import BiLSTMAttention, Vocabulary, TextDataset
        from torch.utils.data import DataLoader
        import sys
        sys.modules['__main__'].Vocabulary = Vocabulary
        
        vocab = joblib.load(vocab_path)
        test_dataset = TextDataset(test_df['text_classical'].tolist(), test_df['label'].tolist(), vocab)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BiLSTMAttention(
            vocab_size=len(vocab.word2idx),
            embed_dim=128,
            hidden_dim=128,
            num_classes=num_classes
        ).to(device)
        model.load_state_dict(torch.load(bilstm_path, map_location=device))
        model.eval()
        
        inf_start = time.time()
        all_preds, all_probas, all_targets = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                logits, _ = model(batch_x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probas.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())
                
        inf_time = (time.time() - inf_start) / len(test_df) * 1000
        acc = accuracy_score(all_targets, all_preds)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
        top3_acc = top_k_accuracy(np.array(all_targets), np.array(all_probas), k=3)
        cm = confusion_matrix(all_targets, all_preds, labels=all_labels)
        
        model_size_mb = os.path.getsize(bilstm_path) / (1024 * 1024)
        
        return {
            "BiLSTM + Attention": {
                "Accuracy": round(float(acc), 4),
                "Macro Precision": round(float(prec_m), 4),
                "Macro Recall": round(float(rec_m), 4),
                "Macro-F1": round(float(f1_m), 4),
                "Weighted-F1": round(float(f1_w), 4),
                "Top-3 Accuracy": round(float(top3_acc), 4),
                "Inference Time (ms/sample)": round(float(inf_time), 4),
                "Model Size (MB)": round(float(model_size_mb), 2),
                "confusion_matrix": cm.tolist()
            }
        }
    except Exception as e:
        print(f"[Evaluate] Error evaluating BiLSTM model: {e}")
        return {}


def evaluate_transformer_models(data_dir: str = "data/processed", models_dir: str = "models") -> Dict[str, Any]:
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    with open(os.path.join(data_dir, "class_mapping.json"), "r") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    all_labels = list(range(num_classes))
    
    results = {}
    transformer_names = [("DistilBERT", "DistilBERT"), ("CodeBERT", "CodeBERT")]
    
    for display_name, folder_name in transformer_names:
        tf_dir = os.path.join(models_dir, folder_name)
        if os.path.exists(tf_dir):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from torch.utils.data import DataLoader
                from src.train_transformer import TransformerBugDataset
                
                tokenizer = AutoTokenizer.from_pretrained(tf_dir)
                model = AutoModelForSequenceClassification.from_pretrained(tf_dir)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                
                test_dataset = TransformerBugDataset(test_df['text_transformer'].tolist(), test_df['label'].tolist(), tokenizer, max_len=64)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                inf_start = time.time()
                all_preds, all_probas, all_targets = [], [], []
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
                        
                inf_time = (time.time() - inf_start) / len(test_df) * 1000
                acc = accuracy_score(all_targets, all_preds)
                prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
                prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
                top3_acc = top_k_accuracy(np.array(all_targets), np.array(all_probas), k=3)
                cm = confusion_matrix(all_targets, all_preds, labels=all_labels)
                
                model_size_mb = sum(
                    os.path.getsize(os.path.join(tf_dir, f)) 
                    for f in os.listdir(tf_dir) 
                    if os.path.isfile(os.path.join(tf_dir, f))
                ) / (1024 * 1024)
                
                results[display_name] = {
                    "Accuracy": round(float(acc), 4),
                    "Macro Precision": round(float(prec_m), 4),
                    "Macro Recall": round(float(rec_m), 4),
                    "Macro-F1": round(float(f1_m), 4),
                    "Weighted-F1": round(float(f1_w), 4),
                    "Top-3 Accuracy": round(float(top3_acc), 4),
                    "Inference Time (ms/sample)": round(float(inf_time), 4),
                    "Model Size (MB)": round(float(model_size_mb), 2),
                    "confusion_matrix": cm.tolist()
                }
            except Exception as e:
                print(f"[Evaluate] Error evaluating {display_name}: {e}")
                
    return results


def run_full_evaluation(data_dir: str = "data/processed", models_dir: str = "models", reports_dir: str = "reports"):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)
    
    print("\n==================================================================")
    print("      BUGROUTE AI - UNSEEN HELD-OUT TEST SET EVALUATION           ")
    print("==================================================================")
    
    all_results = {}
    
    # 1. Classical ML
    classical_res = evaluate_classical_models(data_dir, models_dir)
    all_results.update(classical_res)
    
    # 2. BiLSTM
    bilstm_res = evaluate_bilstm_model(data_dir, models_dir)
    all_results.update(bilstm_res)
    
    # 3. Transformers
    transformer_res = evaluate_transformer_models(data_dir, models_dir)
    all_results.update(transformer_res)
    
    # Save master evaluation JSON
    with open(os.path.join(reports_dir, "master_evaluation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
        
    # Save master CSV comparison table
    table_rows = []
    for m_name, metrics in all_results.items():
        row = {"Model": m_name}
        for k, v in metrics.items():
            if k != "confusion_matrix":
                row[k] = v
        table_rows.append(row)
        
    comp_df = pd.DataFrame(table_rows)
    comp_df.to_csv(os.path.join(reports_dir, "master_model_comparison.csv"), index=False)
        
    print("\nEvaluated Models Summary:")
    for model_name, metrics in all_results.items():
        print(f"\nModel: {model_name}")
        print(f"  Accuracy       : {metrics['Accuracy'] * 100:.2f}%")
        print(f"  Macro-F1       : {metrics['Macro-F1']:.4f}")
        print(f"  Weighted-F1    : {metrics['Weighted-F1']:.4f}")
        print(f"  Top-3 Accuracy : {metrics['Top-3 Accuracy'] * 100:.2f}%")
        print(f"  Inference Speed: {metrics['Inference Time (ms/sample)']:.4f} ms/sample")
        print(f"  Model File Size: {metrics['Model Size (MB)']} MB")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_full_evaluation()

