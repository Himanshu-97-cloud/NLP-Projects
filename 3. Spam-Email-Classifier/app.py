# app.py
import streamlit as st
import joblib
import main

st.title("📧 Spam Email Classifier")

# Load models and vectorizer
tfidf = joblib.load("tfidf.pkl")
models = {
    "Naive Bayes": joblib.load("NaiveBayes.pkl"),
    "Logistic Regression": joblib.load("LogisticRegression.pkl"),
    "SVM": joblib.load("SVM.pkl"),
    "Random Forest": joblib.load("RandomForest.pkl")
}

# Dropdown to select model
model_choice = st.selectbox("Choose a model:", list(models.keys()))

# Text input
user_text = st.text_area("Enter an email text to classify:")

if st.button("Classify"):
    if user_text.strip():
        processed = main.preprocess(user_text)
        text_tfidf = tfidf.transform([processed])
        prediction = models[model_choice].predict(text_tfidf)[0]

        if prediction == 1:
            st.error("🚨 This email is classified as **SPAM**")
        else:
            st.success("✅ This email is classified as **HAM (Not Spam)**")
    else:
        st.warning("Please enter some text first!")
