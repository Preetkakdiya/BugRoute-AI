"""
BugRoute AI — Real-World & Synthetic ML Validation Calculator
Computes exact per-model metrics across all 5 models for:
1. Synthetic Held-Out Test Set (375 samples)
2. Real-World Exploratory Generalization Benchmark (20 samples)

Metrics computed per model:
- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1
- Weighted-F1
- Per-class Precision & Recall
- Confusion Matrices
- Inference Latency (ms/sample)
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import BugRoutePredictor
from src.run_real_world_generalization_test import TEST_CASES

MODELS = ["Linear SVM", "Logistic Regression", "BiLSTM + Attention", "DistilBERT", "CodeBERT"]

def evaluate_all():
    print("="*70)
    print("BUGROUTE AI — COMPREHENSIVE ML VALIDATION (ALL 5 MODELS)")
    print("="*70)
    
    predictor = BugRoutePredictor()
    label_names = predictor.label_names
    label_map = {name: i for i, name in enumerate(label_names)}
    
    # -------------------------------------------------------------------------
    # 1. SYNTHETIC HELD-OUT TEST SET EVALUATION (375 SAMPLES)
    # -------------------------------------------------------------------------
    test_csv = os.path.join("data", "processed", "test.csv")
    synthetic_results = {}
    
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        print(f"\n[Part 1] Synthetic Test Set Evaluation ({len(test_df)} samples)")
        
        titles = test_df["title"].fillna("").values
        descriptions = test_df["description"].fillna("").values
        y_true = test_df["label"].values
        
        for m in MODELS:
            start_t = time.time()
            y_pred = []
            for t, d in zip(titles, descriptions):
                res = predictor.predict(t, d, model_choice=m, allow_fallback=False)
                pred_cls = res.get("predicted_component", "Other")
                y_pred.append(label_map.get(pred_cls, 7))
                
            elapsed_ms = ((time.time() - start_t) / len(test_df)) * 1000.0
            
            acc = accuracy_score(y_true, y_pred)
            prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
            per_class_p, per_class_r, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(label_names)), zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
            
            synthetic_results[m] = {
                "sample_count": len(test_df),
                "accuracy": round(float(acc), 4),
                "macro_precision": round(float(prec_m), 4),
                "macro_recall": round(float(rec_m), 4),
                "macro_f1": round(float(f1_m), 4),
                "weighted_f1": round(float(f1_w), 4),
                "inference_ms": round(float(elapsed_ms), 4),
                "per_class_precision": {label_names[i]: round(float(per_class_p[i]), 4) for i in range(len(label_names))},
                "per_class_recall": {label_names[i]: round(float(per_class_r[i]), 4) for i in range(len(label_names))},
                "confusion_matrix": cm.tolist()
            }
            print(f"  - {m:20s} | Acc: {acc*100:6.2f}% | Macro-F1: {f1_m:.4f} | Latency: {elapsed_ms:.3f}ms")

    # -------------------------------------------------------------------------
    # 2. REAL-WORLD EXPLORATORY BENCHMARK EVALUATION (20 SAMPLES)
    # -------------------------------------------------------------------------
    print(f"\n[Part 2] Real-World Exploratory Benchmark Evaluation ({len(TEST_CASES)} samples)")
    real_results = {}
    
    y_true_real = [label_map[tc["expected_class"]] for tc in TEST_CASES]
    
    for m in MODELS:
        start_t = time.time()
        y_pred_real = []
        for tc in TEST_CASES:
            res = predictor.predict(tc["title"], tc["description"], model_choice=m, allow_fallback=False)
            pred_cls = res.get("predicted_component", "Other")
            y_pred_real.append(label_map.get(pred_cls, 7))
            
        elapsed_ms = ((time.time() - start_t) / len(TEST_CASES)) * 1000.0
        
        acc = accuracy_score(y_true_real, y_pred_real)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true_real, y_pred_real, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true_real, y_pred_real, average="weighted", zero_division=0)
        per_class_p, per_class_r, _, _ = precision_recall_fscore_support(y_true_real, y_pred_real, average=None, labels=range(len(label_names)), zero_division=0)
        cm = confusion_matrix(y_true_real, y_pred_real, labels=range(len(label_names)))
        
        real_results[m] = {
            "sample_count": len(TEST_CASES),
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec_m), 4),
            "macro_recall": round(float(rec_m), 4),
            "macro_f1": round(float(f1_m), 4),
            "weighted_f1": round(float(f1_w), 4),
            "inference_ms": round(float(elapsed_ms), 4),
            "per_class_precision": {label_names[i]: round(float(per_class_p[i]), 4) for i in range(len(label_names))},
            "per_class_recall": {label_names[i]: round(float(per_class_r[i]), 4) for i in range(len(label_names))},
            "confusion_matrix": cm.tolist()
        }
        print(f"  - {m:20s} | Acc: {acc*100:6.2f}% | Macro-F1: {f1_m:.4f} | Latency: {elapsed_ms:.3f}ms")

    # Save to json report artifact
    out_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_classes": 9,
            "classes": label_names
        },
        "synthetic_test_set_375": synthetic_results,
        "real_world_benchmark_20": real_results
    }
    
    out_path = os.path.join("reports", "comprehensive_ml_validation.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved comprehensive validation report to: {out_path}")

if __name__ == "__main__":
    evaluate_all()
