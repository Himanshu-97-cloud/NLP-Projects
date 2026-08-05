import streamlit as st
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

st.title("📊 Word Frequency Analyzer")

# Empty text area for user input
user_text = st.text_area("Enter your text or corpus (one sentence per line):", "")

# Align buttons horizontally
col1, col2, col3 = st.columns(3)

with col1:
    run_bow = st.button("Run Bag of Words")
with col2:
    run_ngrams = st.button("Generate N-grams")
with col3:
    run_freq = st.button("Show Word Frequency")

# Bag of Words
if run_bow:
    if user_text.strip():
        corpus = [line for line in user_text.split("\n") if line.strip()]
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(corpus)
        corpus_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        st.write("### Bag of Words Matrix")
        st.write(corpus_bow)
    else:
        st.warning("Please enter some text first!")

# N-grams
if run_ngrams:
    if user_text.strip():
        tokens = word_tokenize(user_text.lower())
        uni = [" ".join(u) for u in ngrams(tokens, 1)]
        bi = [" ".join(b) for b in ngrams(tokens, 2)]
        tri = [" ".join(t) for t in ngrams(tokens, 3)]

        st.write("### Unigrams")
        st.code(str(uni), language="python")  # code-style output

        st.write("### Bigrams")
        st.code(str(bi), language="python")   # code-style output

        st.write("### Trigrams")
        st.code(str(tri), language="python")  # code-style output
    else:
        st.warning("Please enter some text first!")



# Word Frequency
if run_freq:
    if user_text.strip():
        tokens = [w.lower() for w in word_tokenize(user_text) if w.isalpha()]
        frequency = Counter(tokens)
        words, counts = zip(*frequency.most_common())
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(words, counts, color="skyblue")
        ax.set_title("Word Frequency Visualization")
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.warning("Please enter some text first!")
