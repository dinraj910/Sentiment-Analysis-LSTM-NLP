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
import time
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Configuration ────────────────────────────────────────────────────────

MAX_LEN = 200
THRESHOLD = 0.5

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "lstm_sentiment.h5"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"

EXAMPLES = [
    {"label": "😊 Positive", "key": "pos",
     "text": "This product is amazing and exceeded all my expectations. Highly recommended!"},
    {"label": "😞 Negative", "key": "neg",
     "text": "Terrible quality. It broke after one day. Complete waste of money."},
    {"label": "🤔 Nuanced", "key": "mix",
     "text": "The camera is great but the battery life is really disappointing."},
    {"label": "⭐ Review", "key": "rev",
     "text": "Absolutely love this! The design is sleek and performance is top-notch."},
]


# ── Artifact Loading (cached) ───────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    with open(str(TOKENIZER_PATH), "rb") as f:
        return pickle.load(f)


# ── Inference ────────────────────────────────────────────────────────────

def run_inference(text: str, model, tokenizer) -> dict:
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    start = time.perf_counter()
    prob = float(model.predict(padded, verbose=0)[0][0])
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    label = "POSITIVE" if prob >= THRESHOLD else "NEGATIVE"
    confidence = round(abs(prob - THRESHOLD) * 2, 4)

    if confidence >= 0.80:
        level, level_icon = "Very High", "🟢"
    elif confidence >= 0.60:
        level, level_icon = "High", "🔵"
    elif confidence >= 0.35:
        level, level_icon = "Moderate", "🟡"
    else:
        level, level_icon = "Low", "🟠"

    word_count = len(text.split())
    token_count = sum(1 for t in sequences[0] if t != 0)

    return {
        "label": label,
        "confidence": confidence,
        "level": level,
        "level_icon": level_icon,
        "latency_ms": latency_ms,
        "word_count": word_count,
        "token_count": token_count,
    }


# ═════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Neural Sentiment · LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════
#  FUTURISTIC CSS
# ═════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ─── Import Fonts ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ─── Root Theme ─── */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a2e;
    --bg-card-hover: #222240;
    --accent-purple: #7c3aed;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border-subtle: rgba(148, 163, 184, 0.08);
    --glow-purple: rgba(124, 58, 237, 0.3);
    --glow-blue: rgba(59, 130, 246, 0.3);
    --glow-green: rgba(16, 185, 129, 0.25);
    --glow-red: rgba(239, 68, 68, 0.25);
}

/* ─── Global Reset ─── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.stApp > header { background: transparent !important; }

.block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-purple); border-radius: 3px; }

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
}

/* ─── Hero Header ─── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 0.35rem 1.2rem;
    border-radius: 50px;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #fff 0%, #c4b5fd 40%, #818cf8 70%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem;
    line-height: 1.15;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ─── Glass Cards ─── */
.glass-card {
    background: linear-gradient(135deg, rgba(26,26,46,0.9), rgba(18,18,26,0.95));
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-purple), var(--accent-blue), transparent);
    opacity: 0.6;
}

.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.2);
    box-shadow: 0 8px 40px rgba(124, 58, 237, 0.08);
    transform: translateY(-2px);
}

/* ─── Result Cards ─── */
.result-positive {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(6,182,212,0.05));
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 16px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideIn 0.5s ease-out;
}

.result-positive::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan));
}

.result-negative {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(245,158,11,0.05));
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 16px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideIn 0.5s ease-out;
}

.result-negative::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-red), var(--accent-amber));
}

.result-icon {
    font-size: 2.8rem;
    margin-bottom: 0.3rem;
}

.result-label-text {
    font-family: 'Inter', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 3px;
    margin: 0.3rem 0;
}

.result-pos-text { color: var(--accent-green); }
.result-neg-text { color: var(--accent-red); }

.result-confidence {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
}

/* ─── Confidence Bar ─── */
.confidence-bar-track {
    width: 100%;
    height: 8px;
    background: rgba(148,163,184,0.1);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.8rem;
}

.confidence-bar-fill-green {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan));
    box-shadow: 0 0 12px var(--glow-green);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.confidence-bar-fill-red {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-red), var(--accent-amber));
    box-shadow: 0 0 12px var(--glow-red);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ─── Stat Pills ─── */
.stat-row {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(148,163,184,0.06);
    border: 1px solid var(--border-subtle);
    border-radius: 50px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}

.stat-pill-icon { font-size: 0.85rem; }

/* ─── Example Cards ─── */
.example-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.8rem;
    margin-top: 1rem;
}

.example-card {
    background: rgba(26,26,46,0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    cursor: pointer;
    transition: all 0.25s ease;
}

.example-card:hover {
    border-color: rgba(124, 58, 237, 0.3);
    background: rgba(124, 58, 237, 0.06);
    transform: translateY(-2px);
}

.example-card-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--accent-purple);
    margin-bottom: 0.4rem;
}

.example-card-text {
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ─── Pipeline Section ─── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 0;
}

.pipeline-num {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
}

.pipeline-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.pipeline-text strong {
    color: var(--text-primary);
}

/* ─── Sidebar Model Card ─── */
.sidebar-card {
    background: rgba(26,26,46,0.5);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

.sidebar-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent-purple);
    margin-bottom: 0.8rem;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.82rem;
}

.spec-key { color: var(--text-muted); }
.spec-val { color: var(--text-primary); font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }

/* ─── Text Area ─── */
.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 2px var(--glow-purple) !important;
}

.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ─── Buttons ─── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px var(--glow-purple) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px var(--glow-purple) !important;
}

.stButton > button:not([kind="primary"]) {
    background: rgba(26,26,46,0.6) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    transition: all 0.25s ease !important;
}

.stButton > button:not([kind="primary"]):hover {
    border-color: var(--accent-purple) !important;
    color: var(--text-primary) !important;
    background: rgba(124, 58, 237, 0.08) !important;
}

/* ─── Divider ─── */
hr {
    border-color: var(--border-subtle) !important;
    margin: 1.5rem 0 !important;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
}

/* ─── Section Headers ─── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 0.8rem;
}

.section-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
    font-size: 1rem;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
}

/* ─── Footer ─── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-muted);
    font-size: 0.75rem;
}

.footer-tech {
    display: inline-flex;
    gap: 0.5rem;
    align-items: center;
    margin-top: 0.4rem;
}

.footer-dot {
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: var(--text-muted);
}

/* ─── Animations ─── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 2s infinite;
    margin-right: 0.4rem;
}

/* ─── Metric cards ─── */
.stMetric, [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
}

/* ─── Hide Streamlit defaults ─── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem;">
        <div style="font-size: 2.2rem; margin-bottom: 0.3rem;">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 800; letter-spacing: -0.5px;
                    background: linear-gradient(135deg, #c4b5fd, #818cf8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Neural Sentinel
        </div>
        <div style="font-size: 0.7rem; color: var(--text-muted); letter-spacing: 2px;
                    text-transform: uppercase; margin-top: 0.2rem;">
            LSTM Sentiment Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Architecture Card
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">⚡ Architecture</div>
        <div class="spec-row"><span class="spec-key">Network</span><span class="spec-val">LSTM</span></div>
        <div class="spec-row"><span class="spec-key">Embedding</span><span class="spec-val">128-d</span></div>
        <div class="spec-row"><span class="spec-key">LSTM Units</span><span class="spec-val">128</span></div>
        <div class="spec-row"><span class="spec-key">Dropout</span><span class="spec-val">0.2 + 0.2</span></div>
        <div class="spec-row"><span class="spec-key">Vocab</span><span class="spec-val">20,000</span></div>
        <div class="spec-row"><span class="spec-key">Max Length</span><span class="spec-val">200</span></div>
        <div class="spec-row"><span class="spec-key">Masking</span><span class="spec-val">Zero-mask</span></div>
        <div class="spec-row"><span class="spec-key">Output</span><span class="spec-val">Sigmoid</span></div>
        <div class="spec-row" style="border:none;"><span class="spec-key">Framework</span><span class="spec-val">TF / Keras</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline Card
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">🔄 Inference Pipeline</div>
        <div class="pipeline-step">
            <div class="pipeline-num">1</div>
            <div class="pipeline-text"><strong>Tokenize</strong> — word-level, lowercase</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-num">2</div>
            <div class="pipeline-text"><strong>Encode</strong> — integer sequences</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-num">3</div>
            <div class="pipeline-text"><strong>Pad</strong> — post, 200 tokens</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-num">4</div>
            <div class="pipeline-text"><strong>LSTM</strong> — forward pass</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-num">5</div>
            <div class="pipeline-text"><strong>Classify</strong> — σ ≥ 0.5</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 0.2rem;">
        <span class="pulse-dot"></span>
        <span style="font-size: 0.78rem; color: var(--accent-green); font-weight: 600;">
            Model Loaded · Ready
        </span>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════

# ── Hero Header ──────────────────────────────────────────────────────

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🚀 Deep Learning · NLP · RNN</div>
    <h1 class="hero-title">Sentiment Analysis</h1>
    <p class="hero-subtitle">
        Classify text as <strong>POSITIVE</strong> or <strong>NEGATIVE</strong> using a
        pre-trained LSTM neural network — trained on 240k+ reviews, deployed in real time.
    </p>
</div>
""", unsafe_allow_html=True)

# Load artifacts (cached)
model = load_model()
tokenizer = load_tokenizer()

# Session state
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.analyzed_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

# ── Input Section ────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
    <div class="section-icon">📝</div>
    <div class="section-title">Enter Text</div>
</div>
""", unsafe_allow_html=True)

text_input = st.text_area(
    "input_area",
    height=140,
    placeholder="Type or paste a review, comment, tweet, or any text you want to analyze...",
    label_visibility="collapsed",
)

# ── Analyze Button ───────────────────────────────────────────────────

if st.button("⚡ Analyze Sentiment", type="primary", use_container_width=True):
    if text_input and text_input.strip():
        with st.spinner(""):
            st.session_state.result = run_inference(text_input.strip(), model, tokenizer)
            st.session_state.analyzed_text = text_input.strip()
            # Add to history
            st.session_state.history.insert(0, {
                "text": text_input.strip()[:80] + ("..." if len(text_input.strip()) > 80 else ""),
                "label": st.session_state.result["label"],
            })
            st.session_state.history = st.session_state.history[:10]  # keep last 10
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# ── Quick Examples ───────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
    <div class="section-icon">💡</div>
    <div class="section-title">Quick Examples</div>
</div>
""", unsafe_allow_html=True)

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    with col:
        if st.button(ex["label"], key=f"ex_{ex['key']}", use_container_width=True):
            with st.spinner(""):
                st.session_state.result = run_inference(ex["text"], model, tokenizer)
                st.session_state.analyzed_text = ex["text"]
                st.session_state.history.insert(0, {
                    "text": ex["text"][:80] + ("..." if len(ex["text"]) > 80 else ""),
                    "label": st.session_state.result["label"],
                })
                st.session_state.history = st.session_state.history[:10]

# ═════════════════════════════════════════════════════════════════════════
#  RESULT DISPLAY
# ═════════════════════════════════════════════════════════════════════════

if st.session_state.result is not None:
    r = st.session_state.result
    is_pos = r["label"] == "POSITIVE"

    st.markdown("---")

    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">🎯</div>
        <div class="section-title">Prediction Result</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Result Card ──
    css_class = "result-positive" if is_pos else "result-negative"
    icon = "😊" if is_pos else "😞"
    text_class = "result-pos-text" if is_pos else "result-neg-text"
    bar_class = "confidence-bar-fill-green" if is_pos else "confidence-bar-fill-red"
    bar_pct = max(int(r["confidence"] * 100), 3)

    st.markdown(f"""
    <div class="{css_class}">
        <div style="display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;">
            <div>
                <div class="result-icon">{icon}</div>
                <div class="result-label-text {text_class}">{r['label']}</div>
                <div class="result-confidence">{r['level_icon']} Confidence: {r['level']}</div>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.3rem;">
                    Model Certainty
                </div>
                <div class="confidence-bar-track">
                    <div class="{bar_class}" style="width: {bar_pct}%;"></div>
                </div>
                <div class="stat-row">
                    <span class="stat-pill"><span class="stat-pill-icon">⏱</span> {r['latency_ms']}ms</span>
                    <span class="stat-pill"><span class="stat-pill-icon">📝</span> {r['word_count']} words</span>
                    <span class="stat-pill"><span class="stat-pill-icon">🔤</span> {r['token_count']} tokens</span>
                    <span class="stat-pill"><span class="stat-pill-icon">{r['level_icon']}</span> {r['level']}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Analyzed text ──
    with st.expander("📄 View analyzed text"):
        st.markdown(f"""
        <div style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.7; padding: 0.5rem 0;">
            {st.session_state.analyzed_text}
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  HISTORY
# ═════════════════════════════════════════════════════════════════════════

if st.session_state.history:
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📋</div>
        <div class="section-title">Recent Analyses</div>
    </div>
    """, unsafe_allow_html=True)

    for i, h in enumerate(st.session_state.history[:5]):
        icon = "🟢" if h["label"] == "POSITIVE" else "🔴"
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0.8rem;
                    background: rgba(26,26,46,0.4); border-radius: 8px; margin-bottom: 0.4rem;
                    border: 1px solid var(--border-subtle);">
            <span style="font-size: 0.75rem;">{icon}</span>
            <span style="font-size: 0.82rem; color: var(--text-secondary); flex: 1;
                         overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                {h['text']}
            </span>
            <span style="font-size: 0.72rem; font-weight: 600; color: {'var(--accent-green)' if h['label'] == 'POSITIVE' else 'var(--accent-red)'};
                         font-family: 'JetBrains Mono', monospace;">
                {h['label']}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="app-footer">
    <div>Built with precision for real-time NLP inference</div>
    <div class="footer-tech">
        <span>TensorFlow</span>
        <span class="footer-dot"></span>
        <span>Keras</span>
        <span class="footer-dot"></span>
        <span>Streamlit</span>
        <span class="footer-dot"></span>
        <span>LSTM</span>
    </div>
</div>
""", unsafe_allow_html=True)
