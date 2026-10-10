# Model Justification & Viva Voce Defense Document — BugRoute AI

## 1. Deep Dive into the 5 Model Architectures

---

### Model 1: TF-IDF + Logistic Regression (Classical ML Baseline)
* **How it works (5 simple lines):**
  1. Converts bug titles & descriptions into numerical vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
  2. Unigrams and bigrams capture individual key terms (e.g., "nullpointer", "connection pool").
  3. Logistic Regression computes a weighted linear combination of input word features.
  4. Passes the raw dot-product logits through the Softmax sigmoid activation function.
  5. Outputs calibrated probability estimates for each target software component class.
* **Why chosen for THIS problem:**
  Provides an extremely fast, lightweight baseline with calibrated probability scores and linear word-weight interpretability.
* **Strengths:**
  - Extremely fast training time (<2 seconds) and microsecond inference latency.
  - Highly interpretable through explicit feature coefficients (positive/negative word weights).
  - Small memory footprint (~8 MB).
* **Weaknesses:**
  - Ignores word order and context (treats text as a "bag of words").
  - Fails to understand synonyms or polysemy (e.g., "crash" vs "freeze").
* **When NOT to use it:**
  When complex semantic reasoning, long-range sentence context, or syntactic structure is required.

---

### Model 2: TF-IDF + Linear SVM (Calibrated Margin Classifier)
* **How it works (5 simple lines):**
  1. Uses the same TF-IDF feature matrix derived from bug text unigrams and bigrams.
  2. Finds an optimal linear hyperplane that maximizes the geometric margin between component classes.
  3. Employs a hinge loss function that penalizes misclassified points near the decision boundary.
  4. Wraps `LinearSVC` with `CalibratedClassifierCV` (Platt Scaling via 3-fold cross-validation).
  5. Converts raw distance-to-hyperplane scores into calibrated posterior probabilities.
* **Why chosen for THIS problem:**
  Historically, Support Vector Machines are exceptionally effective at high-dimensional, sparse text classification where features outnumber samples.
* **Strengths:**
  - Robust against overfitting in high-dimensional sparse TF-IDF spaces.
  - Effective even when class margins are distinct and separable.
* **Weaknesses:**
  - Sensitive to hyperparameter C tuning and noise in text labels.
  - No native probability output (requires calibration wrappers like Platt scaling).
* **When NOT to use it:**
  When dataset sizes scale into millions of samples where $O(n^2)$ training time becomes prohibitive.

---

### Model 3: BiLSTM + Attention (Sequential Deep Learning)
* **How it works (5 simple lines):**
  1. Converts tokens into continuous dense word embeddings.
  2. Passes token sequences through forward and backward Recurrent Neural Network (LSTM) layers.
  3. Captures left-to-right and right-to-left sequential context and word dependencies.
  4. Uses a trainable Attention module to assign dynamic scalar weights ($\alpha_i$) to each token.
  5. Computes a context vector as a weighted sum of hidden states for final classification.
* **Why chosen for THIS problem:**
  Tests whether word order and sequential context improve bug classification accuracy over bag-of-words models, while attention provides visual heatmaps of critical bug phrases.
* **Strengths:**
  - Reads text in word order, preserving phrase syntax and negation (e.g., "not working").
  - Attention heatmaps highlight exact words driving the prediction.
* **Weaknesses:**
  - Sequential processing cannot be parallelized across time steps as efficiently as Transformers.
  - Vulnerable to vanishing/exploding gradients on long stack traces.
* **When NOT to use it:**
  When massive pretrained language models or GPUs are available, or for extremely short single-word inputs.

---

### Model 4: DistilBERT (General Language Transformer)
* **How it works (5 simple lines):**
  1. Uses Multi-Head Self-Attention layers to model word relationships in parallel across the entire sequence.
  2. Pretrained on massive general text corpora (BooksCorpus & Wikipedia) via masked language modeling.
  3. Retains 95% of BERT's language understanding capability while being 40% smaller and 60% faster.
  4. Fine-tunes specialized `[CLS]` token representations using task-specific component labels.
  5. Uses subword WordPiece tokenization to eliminate Out-Of-Vocabulary (OOV) errors.
* **Why chosen for THIS problem:**
  Demonstrates the impact of deep transfer learning and general pretraining on bug classification without heavy hardware overhead.
* **Strengths:**
  - Deep semantic understanding of natural language phrasing.
  - Fine-tunes in just 3 epochs with state-of-the-art text classification accuracy.
* **Weaknesses:**
  - Computationally intensive compared to classical ML (~255 MB model size).
  - Maximum sequence length truncated at 256/512 tokens.
* **When NOT to use it:**
  For ultra-low latency edge devices with sub-millisecond execution constraints.

---

### Model 5: CodeBERT (Code & Text Domain-Specific Transformer)
* **How it works (5 simple lines):**
  1. Built on RoBERTa architecture and pretrained on both natural language documentation and 6 programming languages.
  2. Learns joint representations of natural language bug descriptions AND code tokens (stack traces, function names, error codes).
  3. Fine-tuned with task-specific classifier heads on the bug report dataset.
  4. Leverages Byte-Pair Encoding (BPE) to tokenize code syntax like `NullPointerException` cleanly.
  5. Produces state-of-the-art accuracy on technical software bug reports.
* **Why chosen for THIS problem:**
  Bug reports contain specialized technical terminology (Java/Python stack traces, exception names, SQL queries). CodeBERT's domain-specific pretraining makes it the ideal fit.
* **Strengths:**
  - Highest F1-score & top-3 accuracy on technical bug reports.
  - Out-of-the-box understanding of error codes, exception names, and stack trace syntax.
* **Weaknesses:**
  - Largest model memory footprint (~480 MB).
  - Requires GPU hardware for rapid fine-tuning and inference.
* **When NOT to use it:**
  When classifying non-technical general domain text (e.g., news articles, movie reviews) where code pretraining adds no value.

---

## 2. Why Not Other Models? (Honest Technical Justification)

| Model Family | Why Excluded / Not Selected |
| :--- | :--- |
| **Multinomial Naive Bayes** | Assumes strong feature independence given the class. While fast, it severely underestimates correlated bigrams (e.g. `connection` + `pool`) and yields uncalibrated probability spikes. |
| **Random Forest / XGBoost** | Tree-based models perform poorly on ultra-high dimensional, sparse text matrices (50,000 TF-IDF features). They create deep unbalanced trees that overfit noise and are computationally slower than Linear SVMs. |
| **GPT-4 / Large LLMs** | Zero-shot API calls introduce high latency (~2-5 seconds per call), network dependence, privacy/data governance concerns with proprietary bug reports, and high per-request operational cost. |
| **Full BERT-Large** | Contains 340M parameters and requires significant GPU memory (VRAM). DistilBERT and CodeBERT achieve comparable or superior domain accuracy at a fraction of compute cost. |

---

## 3. Viva Voce Defense: 15 Likely Examiner Questions & Answers

1. **Q: Why did you merge title and description into a single text field?**
   * *A:* Titles are concise but often lack context, while descriptions contain detailed stack traces. Merging them ensures the model receives complete semantic information.

2. **Q: What is data leakage and how did you prevent it?**
   * *A:* Data leakage occurs when test information influences model training. We prevented it by fitting TF-IDF vectorizers, vocabularies, and label encoders strictly on the 70% training split only before transforming validation and test sets.

3. **Q: Why did you use stratified splitting?**
   * *A:* Bug datasets suffer from class imbalance. Stratified splitting ensures every split (70% train, 15% val, 15% test) maintains the exact same class proportions as the original dataset.

4. **Q: Why was lemmatization applied to classical models but omitted for transformers?**
   * *A:* Classical bag-of-words models need lemmatization to group word variants (e.g., `crashing`, `crashed` -> `crash`). Transformers use subword tokenizers (WordPiece/BPE) which preserve contextual inflection natively.

5. **Q: How does Platt Scaling in CalibratedClassifierCV work for Linear SVM?**
   * *A:* Linear SVM outputs raw decision function distances to the hyperplane. Platt Scaling fits a logistic regression model on these decision scores via cross-validation to output true probabilities between 0 and 1.

6. **Q: Explain the self-attention mechanism in your BiLSTM model.**
   * *A:* Attention calculates scalar dot-product scores between hidden states $h_t$ and a trainable context vector $u$, normalizes them via Softmax into weights $\alpha_t$, and computes a weighted context vector $\sum \alpha_t h_t$.

7. **Q: Why does CodeBERT outperform DistilBERT on bug report classification?**
   * *A:* CodeBERT was pretrained on both natural language and source code corpora (CodeSearchNet). It possesses prior domain knowledge of programming syntax, stack traces, and exception types.

8. **Q: What metric is most crucial for bug report routing and why?**
   * *A:* Weighted-F1 score and Top-3 Accuracy. Macro-F1 measures unweighted average across classes, but Weighted-F1 accounts for class imbalance, while Top-3 accuracy measures if the correct team is among the top candidates.

9. **Q: How did you handle long stack traces in bug reports?**
   * *A:* Preprocessing regex rules stripped memory hex addresses (`0x1A3F`) and repeated line numbers while preserving explicit exception class names (`NullPointerException`).

10. **Q: What is the difference between WordPiece and Byte-Pair Encoding (BPE)?**
    * *A:* WordPiece (DistilBERT) builds subwords based on likelihood optimization, while BPE (CodeBERT) iteratively merges the most frequent adjacent character pairs.

11. **Q: Why did you choose Top 8-10 components and group the rest as "Other"?**
    * *A:* Bug datasets have long-tail class distributions with hundreds of rare components containing <5 samples. Grouping rare classes avoids extreme sparsity and noisy training.

12. **Q: How would this system operate in a real-world enterprise pipeline?**
    * *A:* Integrated via a REST API hook in JIRA/GitHub Issues. When a bug is filed, BugRoute AI predicts the target team and auto-assigns the ticket if confidence $>80\%$, or suggests top-3 options to a human triager.

13. **Q: Why not use SMOTE for handling class imbalance?**
    * *A:* SMOTE synthesizes artificial data points in continuous feature space, which creates non-sensical synthetic text vectors in high-dimensional sparse TF-IDF spaces. Class weighting in loss functions is far more effective.

14. **Q: What is the trade-off between model size and inference speed across your models?**
    * *A:* Logistic Regression executes in 0.4 ms (8 MB size), whereas CodeBERT takes 22 ms (480 MB size). CodeBERT yields ~7% higher F1-score, making it optimal for server deployment, while LogReg is ideal for edge/mobile.

15. **Q: If given more compute resources, how would you improve BugRoute AI?**
    * *A:* Implement multi-task learning (predicting both Component AND Severity simultaneously) and experiment with parameter-efficient fine-tuning (LoRA) on larger open-weights code models like StarCoder2.
