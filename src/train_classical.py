"""
BugRoute AI - Classical Machine Learning Models Training Module
Trains & evaluates:
1. TF-IDF + Logistic Regression (calibrated probabilities, interpretable weights)
2. TF-IDF + Linear SVM (CalibratedClassifierCV wrapper for true probabilities)
Includes hyperparameter tuning, metrics output, model saving, and feature explainability.
"""

import os
import time
import joblib
import json
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)

def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """Calculates top-k accuracy for multi-class predictions."""
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(correct))


def get_top_features_per_class(vectorizer: TfidfVectorizer, model: LogisticRegression, label_names: list, top_n: int = 15) -> Dict[str, list]:
    """
    Extracts top N highest-weighted word features for each predicted component class.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = {}
    
    if hasattr(model, "coef_"):
        for i, class_name in enumerate(label_names):
            if i < len(model.coef_):
                top_indices = np.argsort(model.coef_[i])[-top_n:][::-1]
                top_features[str(class_name)] = feature_names[top_indices].tolist()
                
    return top_features


def train_classical_models(data_dir: str = "data/processed", models_dir: str = "models", reports_dir: str = "reports"):
    """
    Full training pipeline for Classical ML Models.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load Preprocessed Data
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    with open(os.path.join(data_dir, "class_mapping.json"), "r") as f:
        class_mapping = json.load(f)
    label_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    
    print(f"[Classical ML] Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")
    
    # Fill NaN values if any
    X_train_text = train_df['text_classical'].fillna("").values
    y_train = train_df['label'].values
    
    X_val_text = val_df['text_classical'].fillna("").values
    y_val = val_df['label'].values
    
    X_test_text = test_df['text_classical'].fillna("").values
    y_test = test_df['label'].values
    
    # 2. Vectorization (FIT ONLY ON TRAIN DATA to prevent data leakage!)
    print("[Classical ML] Fitting TF-IDF Vectorizer (unigrams + bigrams, max 50,000 features)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )
    
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)
    
    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    
    results = {}
    
    # -------------------------------------------------------------
    # MODEL 1: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("\n--- Model 1: TF-IDF + Logistic Regression ---")
    start_time = time.time()
    
    lr_base = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    param_grid_lr = {'C': [0.1, 1.0, 10.0]}
    
    grid_lr = GridSearchCV(lr_base, param_grid_lr, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    
    best_lr = grid_lr.best_estimator_
    train_time_lr = time.time() - start_time
    print(f"[LogReg] Best Hyperparameters: {grid_lr.best_params_} (Training Time: {train_time_lr:.2f}s)")
    
    # Inference timing & metrics on Test set
    inf_start = time.time()
    y_pred_lr = best_lr.predict(X_test)
    y_proba_lr = best_lr.predict_proba(X_test)
    inf_time_per_sample_lr = (time.time() - inf_start) / X_test.shape[0] * 1000  # ms per sample
    
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='weighted')
    _, _, f1_macro_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='macro')
    top3_lr = top_k_accuracy(y_test, y_proba_lr, k=3)
    
    joblib.dump(best_lr, os.path.join(models_dir, "logreg_model.joblib"))
    
    # Save Top 15 features per class for Explainability
    explainability_lr = get_top_features_per_class(vectorizer, best_lr, label_names, top_n=15)
    with open(os.path.join(reports_dir, "explainability_logreg.json"), "w") as f:
        json.dump(explainability_lr, f, indent=2)
        
    results["Logistic Regression"] = {
        "Accuracy": round(float(acc_lr), 4),
        "Macro-F1": round(float(f1_macro_lr), 4),
        "Weighted-F1": round(float(f1_lr), 4),
        "Precision": round(float(prec_lr), 4),
        "Recall": round(float(rec_lr), 4),
        "Top-3 Accuracy": round(float(top3_lr), 4),
        "Training Time (s)": round(float(train_time_lr), 2),
        "Inference Time (ms/sample)": round(float(inf_time_per_sample_lr), 4),
        "Model Size (MB)": round(os.path.getsize(os.path.join(models_dir, "logreg_model.joblib")) / (1024 * 1024), 2)
    }
    
    # -------------------------------------------------------------
    # MODEL 2: TF-IDF + Linear SVM (Calibrated for Probabilities)
    # -------------------------------------------------------------
    print("\n--- Model 2: TF-IDF + Linear SVM ---")
    start_time = time.time()
    
    base_svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    # CalibratedClassifierCV wraps LinearSVC to output posterior probabilities via Platt Scaling
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_svm.fit(X_train, y_train)
    
    train_time_svm = time.time() - start_time
    print(f"[Linear SVM] Model Trained & Calibrated (Training Time: {train_time_svm:.2f}s)")
    
    inf_start = time.time()
    y_pred_svm = calibrated_svm.predict(X_test)
    y_proba_svm = calibrated_svm.predict_proba(X_test)
    inf_time_per_sample_svm = (time.time() - inf_start) / X_test.shape[0] * 1000  # ms per sample
    
    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted')
    _, _, f1_macro_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='macro')
    top3_svm = top_k_accuracy(y_test, y_proba_svm, k=3)
    
    joblib.dump(calibrated_svm, os.path.join(models_dir, "svm_model.joblib"))
    
    results["Linear SVM"] = {
        "Accuracy": round(float(acc_svm), 4),
        "Macro-F1": round(float(f1_macro_svm), 4),
        "Weighted-F1": round(float(f1_svm), 4),
        "Precision": round(float(prec_svm), 4),
        "Recall": round(float(rec_svm), 4),
        "Top-3 Accuracy": round(float(top3_svm), 4),
        "Training Time (s)": round(float(train_time_svm), 2),
        "Inference Time (ms/sample)": round(float(inf_time_per_sample_svm), 4),
        "Model Size (MB)": round(os.path.getsize(os.path.join(models_dir, "svm_model.joblib")) / (1024 * 1024), 2)
    }
    
    print("\n--- CLASSICAL MODELS EVALUATION SUMMARY ---")
    res_df = pd.DataFrame(results).T
    print(res_df.to_string())
    
    res_df.to_csv(os.path.join(reports_dir, "classical_results.csv"))
    return results

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_classical_models()
