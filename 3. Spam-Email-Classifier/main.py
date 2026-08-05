# main.py
import pandas as pd
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

nltk.download("punkt")
nltk.download("stopwords")

# Preprocessing function
def preprocess(text):
    lowercase_text = text.lower()
    special_char = re.sub(r"[^a-zA-Z\s]", "", lowercase_text)
    contraction_text = contractions.fix(special_char)
    tokenized_text = word_tokenize(contraction_text)
    stop_words = set(stopwords.words("english"))
    stopwords_text = [word for word in tokenized_text if word not in stop_words]
    final_text = " ".join(stopwords_text)
    return final_text

def train_and_save(path):
    # Load dataset
    df = pd.read_csv(path, encoding="latin-1")
    df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    df = df.rename(columns={"v1": "Category", "v2": "Text"})
    df["Category"] = df["Category"].map({"ham": 0, "spam": 1})
    df["Text"] = df["Text"].apply(preprocess)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Category"], test_size=0.25, random_state=42)

    # TF-IDF
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)

    # Models
    models = {
        "NaiveBayes": BernoulliNB(),
        "LogisticRegression": LogisticRegression(),
        "SVM": SVC(),
        "RandomForest": RandomForestClassifier()
    }

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        joblib.dump(model, f"{name}.pkl")

    joblib.dump(tfidf, "tfidf.pkl")
    print("✅ Models and vectorizer saved successfully.")

# Run training once
if __name__ == "__main__":
    train_and_save("email_spam.csv")
