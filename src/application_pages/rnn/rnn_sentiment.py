# ==============================
# 1. IMPORTS
# ==============================
import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
from pathlib import Path
import speech_recognition as sr
import requests
from dotenv import load_dotenv
import os

# ==============================
# 2. ENV & PATHS
# ==============================
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR.parents[1] / "assets"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# ==============================
# 3. LOAD ASSETS
# ==============================
@st.cache_resource
def load_assets():
    with open(ASSETS_DIR / "word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open(ASSETS_DIR / "config.pkl", "rb") as f:
        config = pickle.load(f)
    return word2idx, config

word2idx, config = load_assets()
vocab_size = config["vocab_size"]
max_len    = config["max_len"]

# ==============================
# 4. MODEL
# ==============================
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)   # bidirectional → *2, raw logits

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        pooled, _ = torch.max(out, dim=1)        # max pooling over time
        out = self.dropout(pooled)
        return self.fc(out).squeeze(1)           # raw logits

@st.cache_resource
def load_model():
    m = RNNModel(vocab_size)
    m.load_state_dict(torch.load(
        ASSETS_DIR / "rnn_model.pth",
        map_location=torch.device("cpu")
    ))
    m.eval()
    return m

model = load_model()

# ==============================
# 5. HELPERS
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

def pad_sequence_manual(seq, max_len):
    """Post-padding to match training (encode_and_pad in train script)."""
    seq = seq[:max_len]
    return seq + [0] * (max_len - len(seq))

def predict_probability(text):
    """Return positive-class probability in [0,1]."""
    cleaned = clean_text(text)
    tokens  = [word2idx.get(w, 1) for w in cleaned.split()]  # 1 = <UNK>
    padded  = pad_sequence_manual(tokens, max_len)
    tensor  = torch.tensor([padded], dtype=torch.long)
    with torch.no_grad():
        logit = model(tensor).item()
        return torch.sigmoid(torch.tensor(logit)).item()

def predict_sentiment(text):
    output = predict_probability(text)
    if output >= 0.5:
        return "Positive", round(output * 100, 1)
    else:
        return "Negative", round((1 - output) * 100, 1)

def get_word_contributions(text):
    """Estimate each word's contribution by leave-one-out probability change."""
    words = text.split()
    if not words:
        return []

    base_prob = predict_probability(text)
    contributions = []
    for i, word in enumerate(words):
        without_word = " ".join(words[:i] + words[i + 1:])
        prob_without = predict_probability(without_word) if without_word.strip() else 0.5
        contribution = base_prob - prob_without
        contributions.append((word, contribution))
    return contributions

def get_word_color(contribution):
    """Map contribution to color: positive -> green, negative -> red, near zero -> tan."""
    mag = min(abs(contribution) / 0.25, 1.0)
    if contribution >= 0.02:
        intensity = int(180 * mag)
        return f"rgb(0, {100 + intensity}, 60)"
    elif contribution <= -0.02:
        intensity = int(180 * mag)
        return f"rgb({150 + intensity//2}, 40, 40)"
    else:
        return "rgb(150, 140, 100)"  # neutral / brownish

def word_sentiment_html(text):
    """Build an HTML string with words colored by sentence-level contribution."""
    items = get_word_contributions(text)
    spans = []
    for word, contribution in items:
        color = get_word_color(contribution)
        if contribution >= 0.02:
            label = "Positive contribution"
            pct = round(min(abs(contribution) * 100, 100), 1)
        elif contribution <= -0.02:
            label = "Negative contribution"
            pct = round(min(abs(contribution) * 100, 100), 1)
        else:
            label = "Neutral contribution"
            pct = round(min(abs(contribution) * 100, 100), 1)
        spans.append(
            f"<span title='{label}: {pct}%' style='"
            f"color:{color};"
            f"font-weight:600;"
            f"padding:1px 2px;"
            f"border-radius:3px;"
            f"cursor:default;"
            f"'>{word}</span>"
        )
    return "<p style='line-height:2; font-size:1rem;'>" + " ".join(spans) + "</p>"

def speech_to_text():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening… speak now.")
            audio = r.listen(source, timeout=8)
        return r.recognize_google(audio)
    except sr.WaitTimeoutError:
        return "No speech detected (timeout)"
    except sr.UnknownValueError:
        return "Could not understand audio"
    except Exception as e:
        return f"Mic error: {e}"

def improve_text_llm(text, api_key):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [{
            "role": "user",
            "content": (
                "Rewrite the following text to sound polished, professional, and clear. "
                "Keep the original meaning. Return only the improved text, no preamble.\n\n"
                f"{text}"
            )
        }],
        "max_tokens": 512,
        "temperature": 0.6,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError:
        return f"API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Request failed: {e}"

# ==============================
# 6. UI
# ==============================
def rnn_sentiment_page():
    st.title("Sentiment Analyzer")

    st.info(
        "This RNN model is trained on the **IMDB movie review dataset** "
        "(50,000 reviews, binary sentiment). Results are most reliable for "
        "English movie/product review-style text.",
    )

    st.divider()

    st.subheader("Input")

    text = st.text_area(
        "Enter your review or opinion:",
        value=st.session_state.get("spoken_text", ""),
        height=160,
        placeholder="Type something or use the microphone…",
    )

    col_analyze, col_speak = st.columns([3, 1])

    with col_analyze:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    with col_speak:
        speak_clicked = st.button("Speak", use_container_width=True)

    if speak_clicked:
        with st.spinner("Listening…"):
            spoken = speech_to_text()
        st.session_state["spoken_text"] = spoken
        st.rerun()

    st.divider()

    if "analysis_ready" not in st.session_state:
        st.session_state["analysis_ready"] = False
    if "analyzed_text" not in st.session_state:
        st.session_state["analyzed_text"] = ""

    if text != st.session_state.get("analyzed_text", ""):
        st.session_state["analysis_ready"] = False

    if analyze_clicked:
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Running inference…"):
                sentiment, confidence = predict_sentiment(text)

            st.session_state["analysis_ready"] = True
            st.session_state["analyzed_text"] = text
            st.session_state["sentiment"] = sentiment
            st.session_state["confidence"] = confidence

    if st.session_state.get("analysis_ready", False):
        st.subheader("Sentiment Analysis")
        sentiment = st.session_state.get("sentiment")
        confidence = st.session_state.get("confidence")

        if sentiment == "Positive":
            st.success(f"**{sentiment}** - {confidence}% confidence")
        else:
            st.error(f"**{sentiment}** - {confidence}% confidence")

        st.progress(confidence / 100)

        with st.expander("Word-level sentiment breakdown - click to expand"):
            st.caption(
                "Each word is colored by its contribution to the full sentence prediction: "
                "**green** = positive leaning, **red** = negative leaning, "
                "**tan** = neutral. Hover over a word to see its exact score."
            )
            st.markdown(word_sentiment_html(st.session_state.get("analyzed_text", "")), unsafe_allow_html=True)

    st.divider()

    st.subheader("Refine with LLM")

    if NVIDIA_API_KEY:
        st.success("NVIDIA API key loaded from `.env`")
        active_key = NVIDIA_API_KEY
    else:
        active_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-…",
            help="Paste your NVIDIA API key here. You can also set NVIDIA_API_KEY in a .env file.",
        )
        if active_key:
            st.success("API key entered")
        else:
            st.error("No API key - enter one above to enable this feature")

    if st.button("Improve Text"):
        if not text.strip():
            st.warning("Enter some text above first.")
        elif not active_key:
            st.warning("Provide an API key to use this feature.")
        else:
            with st.spinner("Calling LLM…"):
                improved = improve_text_llm(text, active_key)
            st.markdown("**Refined text:**")
            st.write(improved)