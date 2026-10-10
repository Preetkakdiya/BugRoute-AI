"""
BugRoute AI - Data Preprocessing & Pipeline Module
Handles text cleaning, tokenization, stopword removal, lemmatization, 
class consolidation, and stratified splitting (70/15/15) without data leakage.
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ensure required NLTK resources are available safely
for resource in ['stopwords', 'wordnet', 'punkt', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning downloading NLTK resource {resource}: {e}")

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {"the", "a", "an", "is", "are", "and", "or", "in", "on", "at", "to", "for", "with", "this", "that"}


def clean_text_classical(text: str) -> str:
    """
    Preprocessing pipeline tailored for Classical ML models (TF-IDF + LogReg / SVM):
    1. Merge title + description (if not already done).
    2. Lowercase.
    3. Remove URLs, HTML tags, hex addresses (0x1A3F).
    4. Preserve exception class names (e.g. NullPointerException).
    5. Tokenize, remove stopwords, and apply Lemmatization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 4. Remove hex memory addresses (e.g. 0x7fff5fbff8f0)
    text = re.sub(r'0x[0-9a-fA-F]+', ' ', text)

    # 5. Remove long numeric sequences / trace IDs
    text = re.sub(r'\b\d{5,}\b', ' ', text)

    # 6. Keep words and exception names (allow letters, numbers, underscores)
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)

    # 7. Stopword removal & Lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 1
    ]

    return " ".join(cleaned_tokens)


def clean_text_transformer(text: str) -> str:
    """
    Preprocessing pipeline tailored for Deep Learning / Transformers (DistilBERT / CodeBERT):
    1. Lowercase / preserve raw casing.
    2. Remove noise (URLs, HTML tags, hex addresses).
    3. NO manual stemming/lemmatization or stopword removal (subword tokenizers handle context).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs & HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)

    # Remove hex addresses & long trace numbers
    text = re.sub(r'0x[0-9a-fA-F]+', ' ', text)
    text = re.sub(r'\b\d{6,}\b', ' ', text)

    # Normalize excessive whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataset(
    df: pd.DataFrame, 
    top_k_components: int = 9,
    data_dir: str = "data/processed",
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Full dataset preprocessing workflow:
    1. Combines 'title' and 'description' into 'full_text'.
    2. Groups infrequent components into 'Other' if outside top K.
    3. Generates cleaned text for classical models and transformer models.
    4. Stratified split: 70% Train, 15% Validation, 15% Test.
    5. Saves processed files and label encoder mappings.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Merge title + description
    df['full_text_raw'] = df['title'].fillna('') + " " + df['description'].fillna('')
    
    # 2. Consolidate component classes (Top K, rest -> 'Other')
    component_counts = df['component'].value_counts()
    top_components = component_counts.nlargest(top_k_components).index.tolist()
    
    df['component_clean'] = df['component'].apply(
        lambda x: x if x in top_components else 'Other'
    )
    
    print(f"[Preprocessing] Consolidated dataset into {len(df['component_clean'].unique())} component classes.")
    
    # 3. Perform text cleaning
    print("[Preprocessing] Applying classical cleaning (tokenization + lemmatization)...")
    df['text_classical'] = df['full_text_raw'].apply(clean_text_classical)
    
    print("[Preprocessing] Applying transformer cleaning (noise removal)...")
    df['text_transformer'] = df['full_text_raw'].apply(clean_text_transformer)
    
    # 4. Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['component_clean'])
    
    # Save label mapping
    class_mapping = {int(idx): str(label) for idx, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(data_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)
    joblib.dump(label_encoder, os.path.join(data_dir, "label_encoder.joblib"))
    
    # 5. Stratified 70% Train / 15% Val / 15% Test split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=random_seed, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_seed, stratify=temp_df['label']
    )
    
    # Data Leakage Integrity Check: Ensure ZERO text string overlap between Train, Val, and Test
    train_texts = set(train_df['full_text_raw'].tolist())
    val_texts = set(val_df['full_text_raw'].tolist())
    test_texts = set(test_df['full_text_raw'].tolist())
    
    overlap_train_test = train_texts.intersection(test_texts)
    overlap_train_val = train_texts.intersection(val_texts)
    
    print(f"[Integrity Check] Train-Test Text Overlap Count: {len(overlap_train_test)}")
    print(f"[Integrity Check] Train-Val Text Overlap Count: {len(overlap_train_val)}")
    assert len(overlap_train_test) == 0, f"Data leakage error! Found {len(overlap_train_test)} duplicate texts between Train and Test sets."
    
    # Save datasets
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
    print(f"[Preprocessing] Splits saved: Train={len(train_df)} (70%), Val={len(val_df)} (15%), Test={len(test_df)} (15%)")
    
    return train_df, val_df, test_df, label_encoder


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import get_or_create_dataset
    
    raw_df = get_or_create_dataset()
    
    # Show before and after cleaning example
    sample_text = raw_df.iloc[0]['title'] + " " + raw_df.iloc[0]['description']
    print("\n--- PREPROCESSING BEFORE & AFTER EXAMPLE ---")
    print(f"RAW TEXT:\n{sample_text}\n")
    print(f"CLASSICAL CLEANED:\n{clean_text_classical(sample_text)}\n")
    print(f"TRANSFORMER CLEANED:\n{clean_text_transformer(sample_text)}\n")
    
    train_df, val_df, test_df, encoder = preprocess_dataset(raw_df)
