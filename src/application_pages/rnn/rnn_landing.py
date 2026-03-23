import streamlit as st

from src.application_pages.rnn.next_word import next_word_page
from src.application_pages.rnn.rnn_sentiment import rnn_sentiment_page


def rnn_application_page():
    st.title("RNN Applications")
    st.info(
        "Choose an RNN application to explore. Each tool demonstrates a different "
        "real-world use case of recurrent neural networks."
    )

    selected_app = st.radio(
        "Select RNN Application",
        ("Sentiment Analyzer", "Next Word Predictor"),
        horizontal=True,
    )

    st.divider()

    if selected_app == "Sentiment Analyzer":
        rnn_sentiment_page()
    else:
        next_word_page()
