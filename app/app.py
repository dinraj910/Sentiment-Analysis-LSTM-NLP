"""
Sentiment Analysis — LSTM Inference Application
================================================
Production Streamlit interface for binary sentiment classification
using a pre-trained LSTM model with word-level tokenization.

Architecture : Embedding(128d) → LSTM(128) → Dense(1, sigmoid)
Tokenizer    : Keras Tokenizer (vocab=20k, OOV="<OOV>")
Input        : Raw text → tokenized → padded to 200 tokens (post)
Output       : POSITIVE / NEGATIVE with confidence indicator
"""

import streamlit as st
import tensorflow as tf
import pickle
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Configuration ────────────────────────────────────────────────────────
# These MUST match training-time settings exactly.

MAX_LEN = 200
THRESHOLD = 0.5

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "lstm_sentiment.h5"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"

EXAMPLES = {
    "Positive": "This product is amazing and exceeded all my expectations. Highly recommended!",
    "Negative": "Terrible quality. It broke after one day. Complete waste of money.",
    "Nuanced": "The camera is great but the battery life is really disappointing.",
}


# ── Artifact Loading (cached) ───────────────────────────────────────────

@st.cache_resource(show_spinner="Loading LSTM model...")
def load_model():
    """Load the trained Keras LSTM model from disk."""
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    """Load the fitted Keras Tokenizer from disk."""
    with open(str(TOKENIZER_PATH), "rb") as f:
        return pickle.load(f)


# ── Inference ────────────────────────────────────────────────────────────

def run_inference(text: str, model, tokenizer) -> dict:
    """
    Full inference pipeline: text → tokenize → pad → predict → label.

    Returns
    -------
    dict with keys:
        label      : "POSITIVE" or "NEGATIVE"
        confidence : float in [0, 1] (distance from decision boundary, normalized)
        level      : qualitative descriptor ("Very High" / "High" / "Moderate" / "Low")
    """
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequences, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    prob = float(model.predict(padded, verbose=0)[0][0])

    label = "POSITIVE" if prob >= THRESHOLD else "NEGATIVE"

    # Confidence: normalized distance from the 0.5 decision boundary → [0, 1]
    confidence = round(abs(prob - THRESHOLD) * 2, 4)

    if confidence >= 0.80:
        level = "Very High"
    elif confidence >= 0.60:
        level = "High"
    elif confidence >= 0.35:
        level = "Moderate"
    else:
        level = "Low"

    return {"label": label, "confidence": confidence, "level": level}


# ── Result Display ───────────────────────────────────────────────────────

def render_result(result: dict, analyzed_text: str):
    """Render the prediction result card and confidence bar."""
    is_pos = result["label"] == "POSITIVE"
    icon = "😊" if is_pos else "😞"
    css = "result-pos" if is_pos else "result-neg"

    st.markdown(
        f"""
        <div class="result-card {css}">
            <div class="result-label">{icon} {result['label']}</div>
            <div class="result-meta">Confidence: {result['level']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(result["confidence"], text=f"Model certainty — {result['level']}")

    with st.expander("📄 Analyzed text"):
        st.write(analyzed_text)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLES
# ═════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LSTM Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .result-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-pos {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
    }
    .result-neg {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
    }
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-meta {
        font-size: 0.92rem;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Model Card
# ═════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🧠 Model Card")

    st.markdown(
        """
| Component | Spec |
|:--|:--|
| Architecture | LSTM (Many-to-One) |
| Embedding | 128-d, trainable |
| LSTM Units | 128 |
| Dropout | 0.2 (+ recurrent) |
| Vocab Size | 20,000 |
| Sequence Length | 200 (post-padded) |
| Masking | Zero-masked embedding |
| Output | Dense(1, sigmoid) |
| Framework | TensorFlow / Keras |
"""
    )

    st.divider()
    st.subheader("⚙️ Inference Pipeline")
    st.markdown(
        """
1. **Tokenize** — word-level, lowercased
2. **Encode** — integer sequences (`<OOV>` for unknowns)
3. **Pad / Truncate** — post, to 200 tokens
4. **Forward pass** — Embedding → LSTM → Dense
5. **Classify** — sigmoid ≥ 0.5 → POSITIVE
"""
    )

    st.divider()
    st.caption("TensorFlow · Keras · Streamlit")


# ═════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════

st.title("📝 Sentiment Analysis")
st.markdown(
    "Classify text as **POSITIVE** or **NEGATIVE** using a pre-trained "
    "LSTM neural network trained on 240 k+ reviews."
)

# Load artifacts early (cached — runs once)
model = load_model()
tokenizer = load_tokenizer()

# ── Session state for persisting results across reruns ────────────────
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.analyzed_text = ""

# ── Text input ────────────────────────────────────────────────────────
text_input = st.text_area(
    "Enter text to analyze:",
    height=130,
    placeholder="Type or paste a review, comment, or any text here...",
)

# ── Analyze button ────────────────────────────────────────────────────
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if text_input and text_input.strip():
        with st.spinner("Running inference..."):
            st.session_state.result = run_inference(text_input.strip(), model, tokenizer)
            st.session_state.analyzed_text = text_input.strip()
    else:
        st.warning("Please enter some text to analyze.")

# ── Quick examples ────────────────────────────────────────────────────
st.divider()
st.markdown("##### 💡 Quick Examples")

cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    with col:
        if st.button(name, key=f"ex_{name}", use_container_width=True, help=text):
            with st.spinner("Running inference..."):
                st.session_state.result = run_inference(text, model, tokenizer)
                st.session_state.analyzed_text = text

# ── Result display (persists via session state) ───────────────────────
if st.session_state.result is not None:
    st.divider()
    render_result(st.session_state.result, st.session_state.analyzed_text)
