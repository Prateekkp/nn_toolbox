import streamlit as st
import torch
import torch.nn as nn
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ASSET_DIR = BASE_DIR / "assets" / "rnn" / "next_word"

# -------------------------------
# Load vocab
# -------------------------------
with open(ASSET_DIR / "vocab.pkl", "rb") as f:
    word_to_idx, idx_to_word = pickle.load(f)

vocab_size = len(word_to_idx)

# -------------------------------
# Model Definition (SAME AS TRAINING)
# -------------------------------
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# -------------------------------
# Load model
# -------------------------------
model = RNNModel(vocab_size, 64, 128)
model.load_state_dict(torch.load(ASSET_DIR / "rnn_wikitext2.pth", map_location="cpu"))
model.eval()

# -------------------------------
# Prediction function
# -------------------------------
sequence_length = 5

def predict_next(text_input):
    words = text_input.lower().split()[-sequence_length:]
    
    # pad if needed
    while len(words) < sequence_length:
        words.insert(0, "<UNK>")
    
    seq = [word_to_idx.get(w, 0) for w in words]
    seq = torch.tensor(seq).unsqueeze(0)
    
    with torch.no_grad():
        output = model(seq)
        predicted_idx = torch.argmax(output, dim=1).item()
    
    return idx_to_word.get(predicted_idx, "<UNK>")

def next_word_page():
    st.title("RNN Next Word Predictor")
    st.info("This model is trained on WikiText-2.")
    st.markdown("Type a sentence and let the RNN predict the next word.")

    user_input = st.text_input("Enter your text:", key="rnn_next_word_input")

    if st.button("Predict Next Word", key="rnn_next_word_button"):
        if user_input.strip() != "":
            prediction = predict_next(user_input)
            st.success(f"Next word: **{prediction}**")
        else:
            st.warning("Please enter some text.")


if __name__ == "__main__":
    st.set_page_config(page_title="RNN Text Predictor", layout="centered")
    next_word_page()