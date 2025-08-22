# streamlit_app.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Optional XAI libs (show helpful messages if not installed)
try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

st.set_page_config(page_title="Identity Deception Detection & Authorship Attribution (with XAI)",
                   layout="wide", initial_sidebar_state="expanded")
st.title("🔍 Identity Deception Detection & Authorship Attribution (with XAI)")

# ---------------------------------------------------------------------
# Feature schema — must match training exactly
# ---------------------------------------------------------------------
NUMERIC_COLS = [
    "char_len","word_count","avg_word_len","type_token_ratio","uppercase_ratio","digit_ratio",
    "exclamation_count","question_count","period_count","comma_count","url_count","mention_count",
    "email_count","currency_count","number_token_count","emoji_count","pronoun_ratio","authenticity_score"
]
CATEGORICAL_COLS = ["channel","intent","tone","writing_style"]
TEXT_COL = "message_clean"

def extract_features(raw: str) -> dict:
    """Return ONE row with all columns expected by the pipeline."""
    s = str(raw)
    s_basic = re.sub(r"\s+", " ", s).strip()
    s_clean = s_basic.lower()

    tokens = s_clean.split()
    n_tokens = len(tokens)

    number_token_count = sum(t.isdigit() for t in tokens)
    currency_count     = s_basic.count("$") + s_basic.count("£") + s_basic.count("€")
    exclamation_count  = s_basic.count("!")
    question_count     = s_basic.count("?")
    period_count       = s_basic.count(".")
    comma_count        = s_basic.count(",")
    url_count          = len(re.findall(r"http[s]?://\S+|www\.\S+", s_basic))
    mention_count      = s_basic.count("@")
    email_count        = len(re.findall(r"\b\S+@\S+\.\S+\b", s_basic))
    emoji_vocab        = "😀😁😂🤣😊😍😢😭👍🙏🔥❤️💯"
    emoji_count        = sum(1 for c in s_basic if c in emoji_vocab)

    avg_word_len       = (sum(len(t) for t in tokens) / n_tokens) if n_tokens else 0.0
    type_token_ratio   = (len(set(tokens)) / n_tokens) if n_tokens else 0.0
    pronoun_ratio      = (sum(t in {
        "i","we","you","he","she","they","me","us","him","her","them",
        "my","our","your","his","her","their"
    } for t in tokens) / n_tokens) if n_tokens else 0.0

    char_len           = len(s_basic)
    uppercase_ratio    = sum(1 for c in s if c.isupper()) / char_len if char_len else 0.0
    digit_ratio        = sum(1 for c in s if c.isdigit()) / char_len if char_len else 0.0

    # Categoricals (the OHE step expects these columns to exist)
    channel        = "Chat"
    intent         = "Unknown"
    tone           = "Neutral"
    writing_style  = "unknown"

    # Extra numeric used during training
    authenticity_score = 0.5
    word_count         = n_tokens

    return {
        # text that feeds TF-IDF
        TEXT_COL: s_clean,
        # categoricals for OHE
        "channel": channel, "intent": intent, "tone": tone, "writing_style": writing_style,
        # numeric passthrough
        "char_len": char_len, "word_count": word_count, "avg_word_len": avg_word_len,
        "type_token_ratio": type_token_ratio, "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio, "exclamation_count": exclamation_count,
        "question_count": question_count, "period_count": period_count, "comma_count": comma_count,
        "url_count": url_count, "mention_count": mention_count, "email_count": email_count,
        "currency_count": currency_count, "number_token_count": number_token_count,
        "emoji_count": emoji_count, "pronoun_ratio": pronoun_ratio,
        "authenticity_score": authenticity_score,
    }

# ---------------------------------------------------------------------
# Load models (authorship optional)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    deception = joblib.load("deception_gbc.joblib")  # required
    # authorship is optional
    try:
        authorship = joblib.load("authorship_rfc.joblib")
        author_le  = joblib.load("author_label_encoder.joblib")
    except Exception:
        authorship, author_le = None, None
    return deception, authorship, author_le

try:
    deception_model, authorship_model, author_le = load_models()
except Exception as e:
    st.error(f"Could not load deception model (deception_gbc.joblib). Details: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Helper: predict_proba using only numeric changes (for SHAP/LIME)
# ---------------------------------------------------------------------
def f_numeric_predict_proba(numeric_array: np.ndarray, base_row: pd.Series) -> np.ndarray:
    """
    Build full rows by replacing only NUMERIC_COLS with numeric_array rows,
    keep TEXT_COL and categoricals constant from base_row; then call the pipeline.
    """
    rows = []
    for vals in np.array(numeric_array):
        full = base_row.copy()
        full[NUMERIC_COLS] = vals
        rows.append(full.to_dict())
    df = pd.DataFrame(rows)
    return deception_model.predict_proba(df)

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold for 'Deceptive'", 0.10, 0.90, 0.50, 0.05)
    shap_nsamples = st.slider("SHAP samples", 50, 200, 100, 10)

tab1, tab2 = st.tabs(["📝 Single Message", "📦 Batch CSV"])

# ---------------------------------------------------------------------
# Single message tab
# ---------------------------------------------------------------------
with tab1:
    txt = st.text_area("Enter a message:", height=160,
                       placeholder="e.g., Over 70 billion goes unclaimed — follow the link to claim!")
    if st.button("Analyze", type="primary"):
        if not txt.strip():
            st.warning("Please enter text")
        else:
            # Build full feature row (all expected columns present)
            full_row = pd.Series(extract_features(txt))
            feats_full = pd.DataFrame([full_row])

            # Predictions
            prob = float(deception_model.predict_proba(feats_full)[0, 1])
            label = "Deceptive" if prob >= threshold else "Genuine"

            if authorship_model is not None and author_le is not None:
                try:
                    a_idx = int(authorship_model.predict(feats_full)[0])
                    author = author_le.inverse_transform([a_idx])[0]
                except Exception as e:
                    author = f"(error: {e})"
            else:
                author = "(authorship model not attached)"

            st.subheader("Results")
            st.write(f"**Deception:** {label} — Probability: **{prob:.3f}** (threshold={threshold:.2f})")
            st.write(f"**Predicted Author:** {author}")

            # ---------------- XAI Tabs ----------------
            xai_tabs = st.tabs(["📊 SHAP (dotted, numeric features)", "🟢 LIME (numeric features)"])

            # SHAP (dotted plot) on numeric subset
            with xai_tabs[0]:
                if shap is None:
                    st.info("Install SHAP to enable this tab: `pip install shap`")
                else:
                    try:
                        x0 = feats_full[NUMERIC_COLS].values  # (1, d)
                        # background for KernelExplainer (add small noise to avoid all-zeros)
                        bg = np.repeat(x0, 60, axis=0)
                        bg = bg + np.random.normal(0, 0.25, bg.shape)

                        def f_shap(X):
                            return f_numeric_predict_proba(np.array(X), full_row)[:, 1]

                        explainer = shap.KernelExplainer(f_shap, bg)
                        sv = explainer.shap_values(x0, nsamples=int(shap_nsamples))
                        sv_arr = np.array(sv)
                        if sv_arr.ndim == 1:
                            sv_arr = sv_arr.reshape(1, -1)

                        plt.figure(figsize=(7, 4))
                        shap.summary_plot(
                            sv_arr,
                            features=x0,
                            feature_names=NUMERIC_COLS,
                            plot_type="dot",
                            show=False
                        )
                        st.pyplot(plt.gcf())
                        plt.close()
                    except Exception as e:
                        st.warning(f"SHAP not available: {e}")

            # LIME (bar chart) on numeric subset
            with xai_tabs[1]:
                if LimeTabularExplainer is None:
                    st.info("Install LIME to enable this tab: `pip install lime`")
                else:
                    try:
                        x0 = feats_full[NUMERIC_COLS].values
                        bg = np.repeat(x0, 120, axis=0)
                        bg = bg + np.random.normal(0, 0.25, bg.shape)

                        explainer = LimeTabularExplainer(
                            training_data=bg,
                            feature_names=NUMERIC_COLS,
                            class_names=["Genuine", "Deceptive"],
                            discretize_continuous=True,
                            verbose=False
                        )

                        def predict_fn(X):
                            return f_numeric_predict_proba(np.array(X), full_row)

                        exp = explainer.explain_instance(
                            data_row=x0[0],
                            predict_fn=predict_fn,
                            num_features=min(10, len(NUMERIC_COLS))
                        )

                        st.write("**LIME Feature Contributions**")
                        st.write(exp.as_list())

                        fig = exp.as_pyplot_figure()
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"LIME not available: {e}")

# ---------------------------------------------------------------------
# Batch CSV tab
# ---------------------------------------------------------------------
with tab2:
    st.write("Upload a CSV with a **text** column (or download the template).")
    template = pd.DataFrame({"text": [
        "Please send the updated report by EOD.",
        "URGENT: your account is locked—verify at http://secure-verify-login.com",
        "Team standup at 10:00 on Zoom.",
        "Your loan of $10,000 has been approved! Contact us immediately.",
        "Don't miss this chance to win a free iPhone. Click now!"
    ]})
    st.download_button("Download CSV template", template.to_csv(index=False), "sample_batch.csv")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df_in = pd.read_csv(up, encoding="latin-1")

        if "text" not in df_in.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            feats_list = [extract_features(t) for t in df_in["text"]]
            feats_full = pd.DataFrame(feats_list)

            probs = deception_model.predict_proba(feats_full)[:, 1]
            labels = np.where(probs >= threshold, "Deceptive", "Genuine")

            out = df_in.copy()
            out["deception_proba"] = probs
            out["deception_label"] = labels

            if authorship_model is not None and author_le is not None:
                try:
                    idxs = authorship_model.predict(feats_full).astype(int)
                    out["predicted_author"] = author_le.inverse_transform(idxs)
                except Exception as e:
                    out["predicted_author"] = f"(error: {e})"

            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(50))
            st.download_button(
                "📥 Download results CSV",
                out.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )

st.caption("Place models in the same folder: deception_gbc.joblib (required), "
           "authorship_rfc.joblib + author_label_encoder.joblib (optional).")
