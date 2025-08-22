#!/usr/bin/env python3
"""
Train deception (GBC) and authorship (RFC) models matching the Streamlit app.
Saves:
- deception_gbc.joblib
- authorship_rfc.joblib
- author_label_encoder.joblib
- metrics.json
"""

import os, re, string, json
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ---------- Feature funcs (mirror the app) ----------
def basic_clean(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_for_tokens(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " MENTION ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def extract_features(raw: str) -> dict:
    s = str(raw)
    s_basic = basic_clean(s)
    s_clean = clean_for_tokens(s_basic)
    tokens = s_clean.split()
    n_tokens = len(tokens)
    unique_tokens = set(tokens) if n_tokens > 0 else set()
    char_len = len(s_basic)
    num_upper = sum(1 for ch in s if ch.isupper())
    num_digits = sum(1 for ch in s if ch.isdigit())
    num_excl = s.count("!")
    num_q = s.count("?")
    num_period = s.count(".")
    num_commas = s.count(",")
    num_urls = len(re.findall(r"http\S+|www\.\S+", s, flags=re.IGNORECASE))
    num_mentions = len(re.findall(r"@\w+", s))
    num_emails = len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s))
    num_currency = len(re.findall(r"[$€£₦]", s))
    num_numbers = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", s))
    num_emoji = len(re.findall(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]", s))
    avg_word_len = (sum(len(t) for t in tokens) / n_tokens) if n_tokens > 0 else 0.0
    ttr = (len(unique_tokens) / n_tokens) if n_tokens > 0 else 0.0
    pronouns = {"i","you","we","they","he","she","me","us","them","him","her","my","your","our","their","mine","yours","ours","theirs"}
    pronoun_count = sum(1 for t in tokens if t in pronouns)
    pronoun_ratio = pronoun_count / n_tokens if n_tokens > 0 else 0.0
    uppercase_ratio = num_upper / char_len if char_len > 0 else 0.0
    digit_ratio = num_digits / char_len if char_len > 0 else 0.0
    return {
        "message": s,
        "message_clean": s_clean,
        "char_len": char_len,
        "word_count": n_tokens,
        "avg_word_len": avg_word_len,
        "type_token_ratio": ttr,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "exclamation_count": num_excl,
        "question_count": num_q,
        "period_count": num_period,
        "comma_count": num_commas,
        "url_count": num_urls,
        "mention_count": num_mentions,
        "email_count": num_emails,
        "currency_count": num_currency,
        "number_token_count": num_numbers,
        "emoji_count": num_emoji,
        "pronoun_ratio": pronoun_ratio,
        "channel": "Chat",
        "intent": "Unknown",
        "tone": "Neutral",
        "writing_style": "unknown"
    }

def load_and_featurize(path):
    df = pd.read_csv(path)
    feats = df["text"].apply(extract_features).apply(pd.Series)
    for col in ["channel","intent","tone","writing_style","author_id"]:
        if col in df.columns:
            feats[col] = df[col]
    feats["authenticity_score"] = 0.5
    feats["label"] = df["label"].map({"genuine":0,"deceptive":1})
    return feats

def main(data_dir="."):
    train_df = load_and_featurize(os.path.join(data_dir, "train.csv"))
    valid_df = load_and_featurize(os.path.join(data_dir, "valid.csv"))
    test_df  = load_and_featurize(os.path.join(data_dir, "test.csv"))

    text_col = "message_clean"
    cat_cols = ["channel","intent","tone","writing_style"]
    num_cols = ["char_len","word_count","avg_word_len","type_token_ratio","uppercase_ratio","digit_ratio",
                "exclamation_count","question_count","period_count","comma_count","url_count","mention_count",
                "email_count","currency_count","number_token_count","emoji_count","pronoun_ratio","authenticity_score"]

    prep_with_num = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95), text_col),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        verbose_feature_names_out=False
    )

    X_tr, y_tr = train_df.drop(columns=["label","author_id"], errors="ignore"), train_df["label"]
    X_va, y_va = valid_df.drop(columns=["label","author_id"], errors="ignore"), valid_df["label"]
    X_te, y_te = test_df.drop(columns=["label","author_id"], errors="ignore"), test_df["label"]

    # ----- Deception: GradientBoostingClassifier (GBC) -----
    deception_pipe = Pipeline([
        ("prep", prep_with_num),
        ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42))
    ])
    deception_pipe.fit(X_tr, y_tr)

    def eval_deception(X, y):
        proba = deception_pipe.predict_proba(X)[:,1]
        preds = (proba >= 0.5).astype(int)
        report = classification_report(y, preds, output_dict=True)
        try:
            roc = roc_auc_score(y, proba)
        except Exception:
            roc = float('nan')
        cm = confusion_matrix(y, preds).tolist()
        return {"report": report, "roc_auc": roc, "confusion_matrix": cm}

    metrics = {
        "valid": eval_deception(X_va, y_va),
        "test": eval_deception(X_te, y_te),
    }

    joblib.dump(deception_pipe, os.path.join(data_dir, "deception_gbc.joblib"))

    # ----- Authorship: RandomForestClassifier (RFC) -----
    le = LabelEncoder()
    y_author_tr = le.fit_transform(train_df["author_id"])
    y_author_te = le.transform(test_df["author_id"])

    authorship_pipe = Pipeline([
        ("prep", prep_with_num),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])
    authorship_pipe.fit(X_tr, y_author_tr)

    author_acc_test = float((authorship_pipe.predict(X_te) == y_author_te).mean())

    joblib.dump(authorship_pipe, os.path.join(data_dir, "authorship_rfc.joblib"))
    joblib.dump(le, os.path.join(data_dir, "author_label_encoder.joblib"))

    metrics["author_test_accuracy"] = author_acc_test
    with open(os.path.join(data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved models and metrics to", data_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=".")
    args = p.parse_args()
    main(args.data_dir)
