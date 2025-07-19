# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
model = tf.keras.models.load_model("next_word_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

total_words = len(tokenizer.word_index) + 1

# Predict next word function
def predict_next_word(seed_text, top_n=1):
    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    token_list = pad_sequences([token_list], maxlen=5, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0]
    top_indices = predicted.argsort()[-top_n:][::-1]
    return [word for word, index in tokenizer.word_index.items() if index in top_indices]

# Streamlit UI
st.title("📝 Next Word Prediction with LSTM")
st.write("Enter a seed sentence and let the model suggest the next word!")

seed_text = st.text_input("Seed Text (at least 5 words):", "")
top_n = st.slider("Show top N predictions:", 1, 5, 3)

if st.button("Predict"):
    if len(seed_text.split()) >= 1:
        suggestions = predict_next_word(seed_text, top_n=top_n)
        st.success(f"Top {top_n} suggestions:")
        for i, word in enumerate(suggestions, 1):
            st.write(f"{i}. **{word}**")
    else:
        st.warning("Please enter at least one word to begin prediction.")
