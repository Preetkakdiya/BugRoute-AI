# BugRoute AI — Unified Master Project Audit & Technical Evaluation Report

> **Project Overview**: BugRoute AI automates software bug triage by classifying issue descriptions and exception stack traces into 9 engineering component teams (`Auth/Security`, `Backend API`, `Build/CI-CD`, `Compiler/Parser`, `Database`, `Memory/Performance`, `Network/HTTP`, `Other`, `UI/UX`). This report consolidates all technical audits, 5-model ML evaluations, 20-sample real-world generalization benchmarks, logger security audits, and automated Playwright browser end-to-end test results.

---

## 📊 1. Empirical Model Evaluation & Generalization Summary

### 1.1 Synthetic Held-Out Test Set (375 Unique Samples)
- **Split Strategy**: Stratified 70/15/15 train/val/test split with seed `42`. Zero text string overlap (`train_test_overlap = 0`).

| Model Architecture | Accuracy | Macro-F1 | Weighted-F1 | Top-3 Accuracy | Inference Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **100.00%** | **1.0000** | **1.0000** | **100.00%** | **0.559 ms** |
| **Linear SVM** | **100.00%** | **1.0000** | **1.0000** | **100.00%** | **9.179 ms** |
| **BiLSTM + Attention** | **100.00%** | **1.0000** | **1.0000** | **100.00%** | **4.552 ms** |
| **CodeBERT** | **66.13%** | **0.6036** | **0.6541** | **84.00%** | **92.557 ms** |
| **DistilBERT** | **64.80%** | **0.6177** | **0.6412** | **89.07%** | **58.206 ms** |

---

### 1.2 Real-World Exploratory Benchmark (20 Non-Synthetic Bug Reports)
- **Dataset Nature**: 20 manually authored, non-synthetic bug reports covering all 9 classes and complex multi-symptom defects.

| Model Architecture | Real-World Exploratory Accuracy | Macro-F1 | Weighted-F1 | Mean Confidence | Generalization Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear SVM (Default)** | **90.00%** (18/20) | **0.9026** | **0.8967** | **0.8447** | 🏆 **Best Real-World Generalization** |
| **Logistic Regression** | **85.00%** (17/20) | **0.8444** | **0.8350** | **0.2021** | 🥈 Robust TF-IDF baseline |
| **DistilBERT** | **60.00%** (12/20) | **0.4772** | **0.5250** | **0.2171** | Moderate transfer; synthetic shift sensitivity |
| **CodeBERT** | **60.00%** (12/20) | **0.4820** | **0.5367** | **0.1835** | Moderate transfer; minor class bias |
| **BiLSTM + Attention** | **35.00%** (7/20) | **0.3651** | **0.3429** | **0.8081** | High OOV degradation; unmasked `<UNK>` bias |

---

## 🛠️ 2. Prediction Logger Security & Concurrency Audit

### Key Security & Concurrency Protections ([prediction_logger.py](file:///c:/Users/Preet/OneDrive/Desktop/NLP%20MINI/src/prediction_logger.py)):
1. **Thread & Process File-Locking**: Re-entrant `threading.Lock()` + `filelock.FileLock()` context managers prevent data corruption across concurrent requests.
2. **Atomic Writes**: Writes to `tempfile.mkstemp()`, flushes, syncs (`os.fsync`), and atomically replaces target file via `os.replace()`.
3. **Collision-Proof Backup Preservation**: Malformed JSON syntax or non-list root objects are preserved to `.corrupted_YYYYMMDD_*` or `.invalid_struct_YYYYMMDD_*` backups. Write aborts if backup fails.
4. **Sensitive Data Sanitization**: Passwords, Bearer tokens, JWT signatures, and API keys are redacted via regex (`password=[REDACTED]`, `Bearer [REDACTED_TOKEN]`).

---

## 🌐 3. Automated Playwright Browser E2E UI Audit

Automated browser testing was executed using **Playwright Headless Chromium** against the live Streamlit web application on `http://localhost:8501`:

```bash
py -3.10 src/test_browser_e2e.py
# Output: Ran 7 tests in 43.349s — OK (100% PASS)
```

| Test ID | Test Scenario | Verified UI Elements | Status |
| :--- | :--- | :--- | :---: |
| `TC-B101` | App Load & Page Title | Page title `"BugRoute AI — Automated Bug Routing"` renders cleanly | **PASS** |
| `TC-B102` | Sidebar Radio Tab Navigation | Clicks all 4 radio tabs (`Classify`, `Compare`, `Insights`, `About`) | **PASS** |
| `TC-B103` | Preset Quick Samples Click | Clicks `"Database Deadlock"`, auto-fills `SQLException` in `#Bug Title` | **PASS** |
| `TC-B104` | Classification Workflow | Fills inputs, clicks button, verifies prediction result card renders | **PASS** |
| `TC-B105` | Empty Input Safety Warning | Clears inputs, clicks classify, verifies warning banner displays | **PASS** |
| `TC-B106` | Prediction Correction Submission | Expands correction section, submits feedback, verifies success banner | **PASS** |
| `TC-B107` | Evaluation Dashboard UI | Navigates to dashboard, verifies synthetic & real-world tables render | **PASS** |

---

## 📋 4. Master Risk & Mitigation Verification Matrix

| Verification Area | Tested Module / Benchmark | Category | Status |
| :--- | :--- | :---: | :---: |
| **Synthetic Test Set Data Leakage** | `train_test_overlap = 0`, fixed seed 42 in `src/preprocess.py` | ML Integrity | **PASS** |
| **Preprocessing Parity** | `clean_text_classical()` & `clean_text_transformer()` in `src/predict.py` | ML Parity | **PASS** |
| **BiLSTM OOV `<UNK>` Handling** | Analyzed in `src/train_bilstm.py`; 35% exploratory accuracy documented | ML Validation | **PASS (Documented)** |
| **Real-World SVM Generalization** | Evaluated on 20 non-synthetic bug reports (90.00% accuracy) | ML Validation | **PASS** |
| **Logger Thread & Process Lock** | `FileLock` + `_THREAD_LOCK` in `src/prediction_logger.py` | Logger Security | **PASS** |
| **Atomic Write & Backup Abort** | `os.replace` & backup failure write abort logic | Logger Security | **PASS** |
| **Sensitive Credential Redaction** | Regex sanitization for passwords, Bearer tokens, JWTs | Security Audit | **PASS** |
| **Live Browser UI App Load** | `test_browser_101_app_load_and_title` via Playwright Chromium | Browser E2E | **PASS** |
| **Live Browser UI Tab Navigation** | `test_browser_102_navigation_tabs` via Playwright Chromium | Browser E2E | **PASS** |
| **Live Browser Classification Workflow** | `test_browser_104_classification_execution_linear_svm` via Playwright | Browser E2E | **PASS** |
| **Live Browser Correction Submission** | `test_browser_106_prediction_correction_form_rendering` via Playwright | Browser E2E | **PASS** |
| **Headless E2E Acceptance Suite** | `py -3.10 src/test_e2e_acceptance.py` (22 / 22 tests OK) | Integration Suite | **PASS** |

---

## 🛑 5. Project Readiness Verdict

- **Model Readiness Verdict**: **PROVISIONAL / RECOMMENDED FOR HUMAN-IN-THE-LOOP TRIAGE ONLY**. Linear SVM is set as the default model choice with mandatory low-confidence warnings (< 50%).
- **Code & Test Suite Health**: 100% PASS across unit, integration, and Playwright browser E2E test suites.
