import argparse, re, os, numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ---------- MUST match your training/app ----------
NUMERIC_COLS = [
    "char_len","word_count","avg_word_len","type_token_ratio","uppercase_ratio","digit_ratio",
    "exclamation_count","question_count","period_count","comma_count","url_count","mention_count",
    "email_count","currency_count","number_token_count","emoji_count","pronoun_ratio","authenticity_score"
]
CATEGORICAL_COLS = ["channel","intent","tone","writing_style"]
TEXT_COL = "message_clean"

def extract_features(raw: str) -> dict:
    s = str(raw); s_basic = re.sub(r"\s+", " ", s).strip(); s_clean = s_basic.lower()
    tokens = s_clean.split(); n = len(tokens)
    number_token_count = sum(t.isdigit() for t in tokens)
    currency_count = s_basic.count("$")+s_basic.count("£")+s_basic.count("€")
    exclamation_count, question_count = s_basic.count("!"), s_basic.count("?")
    period_count, comma_count = s_basic.count("."), s_basic.count(",")
    url_count = len(re.findall(r"http[s]?://\S+|www\.\S+", s_basic))
    mention_count, email_count = s_basic.count("@"), len(re.findall(r"\b\S+@\S+\.\S+\b", s_basic))
    emoji_count = sum(1 for c in s_basic if c in "😀😁😂🤣😊😍😢😭👍🙏🔥❤️💯")
    avg_word_len = (sum(len(t) for t in tokens)/n) if n else 0.0
    type_token_ratio = (len(set(tokens))/n) if n else 0.0
    pronoun_ratio = (sum(t in {"i","we","you","he","she","they","me","us","him","her","them","my","our","your","his","her","their"} for t in tokens)/n) if n else 0.0
    char_len = len(s_basic)
    uppercase_ratio = (sum(1 for c in s if c.isupper())/char_len) if char_len else 0.0
    digit_ratio = (sum(1 for c in s if c.isdigit())/char_len) if char_len else 0.0
    # Categorical defaults (keep consistent with training)
    return {
        TEXT_COL: s_clean,
        "channel":"Chat","intent":"Unknown","tone":"Neutral","writing_style":"unknown",
        "char_len":char_len,"word_count":n,"avg_word_len":avg_word_len,"type_token_ratio":type_token_ratio,
        "uppercase_ratio":uppercase_ratio,"digit_ratio":digit_ratio,"exclamation_count":exclamation_count,
        "question_count":question_count,"period_count":period_count,"comma_count":comma_count,"url_count":url_count,
        "mention_count":mention_count,"email_count":email_count,"currency_count":currency_count,
        "number_token_count":number_token_count,"emoji_count":emoji_count,"pronoun_ratio":pronoun_ratio,
        "authenticity_score":0.5
    }

def build_feature_frame(text_series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([extract_features(t) for t in text_series.astype(str)])

def normalize_labels(series: pd.Series) -> pd.Series:
    """Map common string labels to 0/1; pass through numeric."""
    if series.dtype != "object":
        return series.astype(int)
    mapping = {
        "genuine":0,"nondeceptive":0,"non-deceptive":0,"legit":0,"legitimate":0,"ham":0,"negative":0,"0":0,
        "deceptive":1,"deceit":1,"fraud":1,"spam":1,"phishing":1,"positive":1,"1":1
    }
    y = series.astype(str).str.strip().str.lower().map(mapping)
    if y.isna().any():
        bad = series[y.isna()].unique()[:10]
        raise ValueError(f"Unrecognized label values: {bad}. Provide numeric 0/1 or map them in 'normalize_labels'.")
    return y.astype(int)

def main(data_path="train.csv", model_path="deception_gbc.joblib",
         text_col="text", label_col="label", test_size=0.2,
         threshold=0.50, out_dir="fp_analysis", explain_top=5):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    X_text = df[text_col].astype(str)
    y_true  = normalize_labels(df[label_col])

    X_full = build_feature_frame(X_text)

    X_tr, X_te, y_tr, y_te, txt_tr, txt_te = train_test_split(
        X_full, y_true, X_text, test_size=test_size, random_state=42, stratify=y_true
    )

    pipe = joblib.load(model_path)
    proba = pipe.predict_proba(X_te)[:,1]
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    print(f"\nConfusion Matrix @ threshold={threshold:.2f}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"False Positive Rate = {fp/(fp+tn+1e-9):.3f}")
    print("\nClassification report:")
    print(classification_report(y_te, y_pred, target_names=["Genuine","Deceptive"], digits=3))

    eval_df = pd.DataFrame({"text":txt_te.values,"true":y_te.values,"proba_deceptive":proba,"pred":y_pred})
    fps = eval_df[(eval_df["true"]==0) & (eval_df["pred"]==1)].sort_values("proba_deceptive", ascending=False)
    fps.to_csv(os.path.join(out_dir,"false_positives.csv"), index=False, encoding="utf-8")
    print(f"Saved {len(fps)} false positives -> {os.path.join(out_dir,'false_positives.csv')}")

    # Token contrasts (FP vs TN)
    def top_words(series,k=30):
        toks=[]; 
        for t in series: 
            t=re.sub(r"[^a-z0-9@#:/._-]+"," ",t.lower()); toks.extend(t.split())
        return Counter(toks).most_common(k)
    tns = eval_df[(eval_df["true"]==0) & (eval_df["pred"]==0)]
    pd.DataFrame(top_words(fps["text"]), columns=["token","count"]).to_csv(os.path.join(out_dir,"fp_top_tokens.csv"), index=False)
    pd.DataFrame(top_words(tns["text"]), columns=["token","count"]).to_csv(os.path.join(out_dir,"tn_top_tokens.csv"), index=False)

    # Optional SHAP (numeric channel)
    try:
        import shap
        print("\nGenerating SHAP for top false positives (numeric features only)...")
        top_samples = fps.head(explain_top)
        def predict_fn_numeric(X_numeric, base_row):
            rows=[]
            for row_vals in np.array(X_numeric):
                full=base_row.copy()
                for i,col in enumerate(NUMERIC_COLS): full[col]=row_vals[i]
                rows.append(full.to_dict())
            return pipe.predict_proba(pd.DataFrame(rows))[:,1]
        for idx,row in top_samples.iterrows():
            base = pd.Series(extract_features(row["text"]))
            x0 = np.array([base[NUMERIC_COLS].values], dtype=float)
            bg = np.repeat(x0,60,axis=0)+np.random.normal(0,0.25,size=(60,x0.shape[1]))
            explainer = shap.KernelExplainer(lambda X: predict_fn_numeric(X, base), bg)
            sv = explainer.shap_values(x0, nsamples=100)
            vals = np.array(sv).reshape(-1)
            pd.DataFrame({"feature":NUMERIC_COLS,"shap_value":vals,"value":x0[0]}).sort_values(
                "shap_value", key=np.abs, ascending=False).to_csv(os.path.join(out_dir,f"fp_{idx}_shap_numeric.csv"), index=False)
    except Exception as e:
        print("(SHAP skipped) ->", e)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="False Positive analysis for deception detector")
    p.add_argument("--data", default="train.csv"); p.add_argument("--model", default="deception_gbc.joblib")
    p.add_argument("--text_col", default="text"); p.add_argument("--label_col", default="label")
    p.add_argument("--test_size", type=float, default=0.2); p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--out_dir", default="fp_analysis"); p.add_argument("--explain_top", type=int, default=5)
    args = p.parse_args()
    main(args.data, args.model, args.text_col, args.label_col, args.test_size, args.threshold, args.out_dir, args.explain_top)
