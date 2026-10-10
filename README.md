# 🐛 BugRoute AI — Automatic Software Bug Classification & Routing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**BugRoute AI** is an end-to-end Natural Language Processing (NLP) & Deep Learning platform designed to automatically analyze software bug reports (titles, descriptions, log files, and stack traces) and route them to the responsible engineering component team with a confidence score.

---

### 🎓 Academic & Team Information

- **College:** GH Patel College of Engineering and Technology
- **Department:** Computer Engineering Department
- **Branch:** Computer Science and Design
- **Project Title:** BugRoute AI — Intelligent Bug Detection Using NLP
- **Team Members:**
  - **Preet Kakdiya** — `12302130501043`
  - **Swayam Raiyani** — `12302130501053`
  - **Vruti Thakkar** — `12302130501063`

---

## 🏗️ System Architecture

```text
                                  [ Input Bug Report / Log File ]
                                 (Title + Description / .log upload)
                                                  │
                                                  ▼
                                     [ Multiline Log Parser ]
                                  (Groups Stack Traces into Issues)
                                                  │
                                                  ▼
                                      [ Preprocessing Engine ]
                        ┌─────────────────────────┴─────────────────────────┐
                        │                                                   │
           [ Classical Preprocessing ]                         [ Transformer Preprocessing ]
         (Lower, URLs, Stopwords, Lemmatize)                 (Noise Removal, Raw Subwords preserved)
                        │                                                   │
         ┌──────────────┴──────────────┐                         ┌──────────┴──────────┐
         ▼                             ▼                         ▼                     ▼
  [ Model 1: LogReg ]         [ Model 2: LinSVM ]       [ Model 4: DistilBERT ] [ Model 5: CodeBERT ]
  (TF-IDF + Softmax)          (Platt Calibrated Margin) (General Transformer) (Code Domain Transformer)
         │                             │                         │                     │
         └──────────────┬──────────────┴─────────────────────────┴─────────────────────┘
                        │
                        ▼
            [ Model 3: BiLSTM + Attention ]
            (Sequential + Word Heatmap)
                        │
                        ▼
         [ Streamlit Web Application & API ]
         (Real-time Routing & Multi-Issue Log UI)
                        │
                        ▼
         [ Atomic Process/Thread Locked Logger ]
         (JSON Audit Log in user_prediction_corrections.json)
```

---

## 📊 Model Evaluation & Benchmarks Summary

Evaluated across **2,500 synthetic bug reports** (70% Train, 15% Validation, 15% Holdout Test) and an **Exploratory Real-World Benchmark** (20 non-synthetic, out-of-domain software bug reports):

| Model Architecture | Model Category | Synthetic Test Accuracy | Real-World Exploratory Accuracy | Generalization Profile |
| :--- | :--- | :---: | :---: | :--- |
| **Linear SVM** *(Default)* | Classical ML Margin | **100.00%** | **90.00% (18/20)** | 🏆 **Best Real-World Generalization** |
| **Logistic Regression** | Classical ML Baseline | **100.00%** | **85.00% (17/20)** | 🥈 Robust TF-IDF Baseline |
| **DistilBERT** | Pretrained Transformer | **64.80%** | **60.00% (12/20)** | Moderate Transfer |
| **CodeBERT** | Pretrained Transformer | **66.13%** | **60.00% (12/20)** | Moderate Transfer |
| **BiLSTM + Attention** | Sequential Deep Learning | **100.00%** | **35.00% (7/20)** | Overfitting / High OOV Degradation |

> *Note: Real-world evaluation used 20 manually authored out-of-domain bug reports and is exploratory. A larger, independently labeled dataset is required to establish production accuracy.*

---

## ✨ Key Features & Engineering Highlights

1. **5 ML Model Architectures**: Compare predictions across Logistic Regression, Linear SVM, BiLSTM + Attention, DistilBERT, and CodeBERT.
2. **Multi-Issue Log Parser (`src/log_parser.py`)**: Automatically parses `.log` files containing multiple error messages and multiline stack traces into distinct issue cards.
3. **Atomic Thread & Process-Locked Logger (`src/prediction_logger.py`)**: Uses `threading.Lock()` and `filelock.FileLock()` with atomic temporary swaps (`os.replace()`) and sensitive token redaction (`reports/user_prediction_corrections.json`).
4. **Print-Ready A2 Academic Poster (`BugRoute_AI_Academic_Poster_A2.pdf`)**: ISO A2 Landscape poster (594 × 420 mm) at 300 DPI for conference/academic presentation.
5. **Unified Test Suite (`src/test_suite.py`)**: 100% test pass rate across unit, integration, and logger security tests.

---

## 📁 Repository Directory Structure

```text
NLP MINI/
├── .streamlit/
│   └── config.toml                        # Streamlit server & theme config
├── app/
│   └── app.py                             # Streamlit Interactive Web Application
├── data/
│   ├── processed/                         # Standardized train/val/test CSVs & label encoder
│   └── raw/                               # Eclipse & Mozilla bug dataset
├── models/
│   ├── bilstm_model.pt & vocab            # BiLSTM PyTorch weights & vocabulary
│   ├── logreg_model.joblib & vectorizer   # Logistic Regression model checkpoint
│   ├── svm_model.joblib                   # Linear SVM model checkpoint
│   ├── CodeBERT/                          # Fine-tuned CodeBERT transformer model
│   └── DistilBERT/                        # Fine-tuned DistilBERT transformer model
├── reports/
│   ├── MASTER_PROJECT_REPORT.md           # Master technical evaluation & security report
│   ├── BugRoute_AI_Academic_Poster_A2_Preview.png # A2 Poster PNG preview
│   ├── comprehensive_ml_validation.json  # Model metrics dataset
│   ├── explainability_logreg.json        # Feature weight explainability JSON
│   ├── master_evaluation.json            # Evaluation metrics JSON
│   ├── master_model_comparison.csv       # Model metrics comparison CSV
│   ├── real_world_generalization_test.json# Real-world out-of-domain benchmark dataset
│   ├── user_prediction_corrections.json  # Thread & process-locked prediction feedback log
│   └── figures/                          # Poster screenshots & charts
├── src/
│   ├── __init__.py
│   ├── api.py                             # FastAPI REST API endpoints
│   ├── data_loader.py                     # Data loading & preprocessing pipelines
│   ├── evaluate.py                        # Core ML evaluation & metrics generator
│   ├── evaluate_real_world_validation.py # Real-world out-of-domain validation runner
│   ├── log_parser.py                      # Multiline log issue extraction engine
│   ├── predict.py                         # Single-instance & bulk prediction engine
│   ├── prediction_logger.py              # Thread & process-locked logger
│   ├── preprocess.py                      # Text cleaning & tokenization engine
│   ├── test_suite.py                      # Unified project test suite
│   ├── train_bilstm.py                    # BiLSTM PyTorch trainer
│   ├── train_classical.py                 # Classical ML models trainer
│   └── train_transformer.py               # Transformer fine-tuning script
├── BugRoute_AI_Academic_Poster_A2.pdf     # Print-Ready A2 Academic Poster PDF (594x420 mm)
├── MODEL_JUSTIFICATION.md                 # Model defense document & Viva Q&A
├── PRESENTATION_OUTLINE.md                # 12-Slide presentation outline
├── README.md                              # Main project documentation
└── requirements.txt                       # Python dependencies
```

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Virtual Environment
```powershell
# Create & activate virtual environment
py -3.10 -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
py -3.10 src/test_suite.py
```

### 3. Launch Streamlit Web Application
```powershell
streamlit run app/app.py
```
Open browser at `http://localhost:8501`.
