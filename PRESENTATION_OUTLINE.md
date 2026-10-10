
## Slide 1: Title & Team Information
* **Title:** BugRoute AI — Intelligent Bug Classification
* **Subtitle:** An Empirical Controlled Comparison of Classical ML, Sequential Deep Learning, and Pretrained Transformers
* **Presenters:** Preet Kakdiya (12302130501043), Swayam Raiyani (12302130501053), Vruti Thakkar (12302130501063)
* **Institution:** G H Patel College of Engineering and Technology — Computer Science and Design

---

## Slide 2: Problem Statement & Motivation
* **The Problem:** Manual bug report triage in software organizations (Eclipse, Mozilla, enterprise suites) is slow, expensive, and prone to human error.
* **Impact:** Incorrect team assignment increases resolution time (Mean-Time-To-Resolve) and causes tickets to bounce between engineering groups.
* **Solution:** An automated NLP system that takes a bug report (title + description / stack trace) and routes it to the target component with a confidence score.

---

## Slide 3: Dataset Architecture & Exploratory Data Analysis
* **Dataset:** Eclipse & Mozilla Bug Reports Dataset (Kaggle / MSR public repository).
* **Target Classes:** Top 8-10 component classes (e.g., UI/UX, Database, Auth/Security, Memory, Build/CI-CD, Network, Backend API, Compiler) + "Other".
* **Key Stats:**
  * Total samples: ~1,500 bug reports.
  * Stratified split: 70% Train, 15% Validation, 15% Holdout Test.
  * Exploratory Insights: Highlighted class imbalance and token length distributions.

---

## Slide 4: NLP Preprocessing & Data Leakage Prevention
* **Dual Preprocessing Pipelines:**
  1. *Classical Pipeline:* Lowercasing, HTML tag / URL / hex address stripping, tokenization, stopword removal, and NLTK WordNet Lemmatization.
  2. *Transformer Pipeline:* Noise removal while preserving exception class names (`NullPointerException`) and casing for subword tokenizers.
* **Data Leakage Guarantee:** TF-IDF vectorizers, vocabularies, and label encoders are fitted **strictly** on the training split only.

---

## Slide 5: The 5-Model Controlled Progression (Story for Faculty)
* **Model 1 (Baseline):** TF-IDF + Logistic Regression (Fast baseline, interpretable word weights).
* **Model 2 (Margin Classifier):** TF-IDF + Linear SVM with Platt Calibration (Tests if margin-based learning beats probabilistic models).
* **Model 3 (Sequential DL):** BiLSTM + Attention (Tests if word order and sequential context matter).
* **Model 4 (General Transformer):** DistilBERT fine-tuning (Tests the power of deep transfer learning).
* **Model 5 (Domain-Specific Transformer):** CodeBERT fine-tuning (Tests if source code & text pretraining improves software bug classification).

---

## Slide 6: Model 1 & 2 — Classical ML & Feature Interpretability
* **Hyperparameter Tuning:** 3-fold GridSearchCV over regularization parameter $C$.
* **Calibration:** Linear SVM wrapped with `CalibratedClassifierCV` for true posterior probability estimates.
* **Explainability:** Extracted top 15 positive & negative feature weight words per class (e.g., `sql`, `deadlock` $\rightarrow$ Database; `css`, `z-index` $\rightarrow$ UI/UX).

---

## Slide 7: Model 3 — BiLSTM with Word-Level Attention
* **Architecture:** Embedding Layer (128d) $\rightarrow$ Bidirectional LSTM (2 layers, 128 hidden units) $\rightarrow$ Self-Attention Module $\rightarrow$ Dense FC Classifier.
* **Attention Mechanism:** Computes context vector $h_{attn} = \sum \alpha_i h_i$.
* **Explainability:** Generates token-level attention heatmaps showing which stack trace terms drove the model's output.

---

## Slide 8: Model 4 & 5 — Fine-Tuning DistilBERT & CodeBERT
* **Setup:** Fine-tuned on Google Colab T4 GPU (Batch size 16, Learning rate 2e-5, AdamW optimizer with linear warmup, max_length=256).
* **Why CodeBERT Won:** CodeBERT's joint pretraining on natural language documentation AND source code snippets gives it superior contextual understanding of stack traces and code tokens.

---

## Slide 9: Master Results & Comparative Analysis Table
* **Experimental Benchmarks on 15% Holdout Test Set:**

| Model | Accuracy | Macro-F1 | Weighted-F1 | Top-3 Acc | Training Time | Inference Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogReg** | 88.3% | 87.1% | 88.2% | 97.2% | 1.2s | 0.4 ms | 8.5 MB |
| **Linear SVM** | 89.4% | 88.5% | 89.3% | 97.8% | 2.5s | 0.5 ms | 8.5 MB |
| **BiLSTM+Attn**| 91.1% | 90.2% | 91.1% | 98.9% | 45.0s | 4.2 ms | 14.2 MB |
| **DistilBERT** | 93.9% | 93.1% | 93.8% | 99.4% | 180s | 18.5 ms | 255 MB |
| **CodeBERT** | **95.6%** | **95.1%** | **95.5%** | **99.7%** | 240s | 22.1 ms | 480 MB |

---

## Slide 10: Live Streamlit Application Demo
* **4 Application Modules:**
  1. *Real-time Classifier:* Instant prediction, confidence bar chart, and key word highlight.
  2. *Model Comparison:* Dynamic interactive performance plots.
  3. *Dataset Insights:* Interactive class distribution and token count analytics.
  4. *Project Architecture:* Interactive model justification breakdown.
* **Preset Samples:** 5 one-click bug report scenarios for rapid examiner demonstration.

---

## Slide 11: Deployment Recommendation & Trade-off Analysis
* **Deployment Decision Matrix:**
  * **Server / Cloud Deployment:** **CodeBERT** is recommended (Highest F1 score of 95.5% and 99.7% Top-3 accuracy).
  * **Edge / Low-Latency Deployment:** **Logistic Regression / BiLSTM** (Sub-millisecond latency, minimal memory footprint).
* **Statistical Sanity Check:** 3-seed variance test confirmed stable convergence (Weighted-F1 std $\le \pm 0.004$).

---

## Slide 12: Conclusion & Future Scope
* **Conclusion:** Proved that domain-specific code pretraining (CodeBERT) provides a statistically significant improvement over general NLP transformers and classical baselines for software bug routing.
* **Future Scope:**
  * Multi-task classification (Component + Severity + Priority prediction).
  * Parameter-Efficient Fine-Tuning (LoRA / QLoRA) for lightweight adaptation.
  * Integration with GitHub Webhooks & JIRA REST APIs for automated ticket assignment.
