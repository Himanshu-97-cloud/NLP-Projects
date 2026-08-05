# 📧 Spam Email Classifier

A Machine Learning project that classifies an email as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and a Support Vector Machine (SVM) classifier.

---

## 📌 Project Overview

This project demonstrates a complete NLP workflow for spam email classification.

The dataset is preprocessed using common NLP techniques, converted into numerical features using **TF-IDF Vectorization**, and then classified using a **Support Vector Machine (SVM)** model.

A simple **Streamlit web application** is included, allowing users to enter any email message and instantly check whether it is spam or not.

---

## 🚀 Features

* Email text preprocessing
* Text cleaning and normalization
* Stopword removal
* TF-IDF Vectorization
* Spam prediction using SVM
* Interactive Streamlit web application
* Saved trained model for fast predictions

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Contractions
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
Spam-Email-Classifier/
│
├── app.py                  # Streamlit application
├── main.py                 # Prediction functions
├── train_model.py          # Train and save the model
├── email_spam.csv          # Dataset
├── svm_model.pkl           # Trained SVM model
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── requirements.txt
├── README.md
└── Spam_Email_Classifier.ipynb
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Himanshu-97-cloud/NLP-Projects.git
```

Go to the project folder

```bash
cd NLP-Projects/Spam-Email-Classifier
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run the following command once to train the model and save it.

```bash
python train_model.py
```

This will generate:

* `svm_model.pkl`
* `tfidf_vectorizer.pkl`

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧠 NLP Pipeline

1. Load Dataset
2. Clean Dataset
3. Convert labels to numerical values
4. Lowercase text
5. Remove special characters
6. Expand contractions
7. Tokenize text
8. Remove stopwords
9. Convert text into TF-IDF vectors
10. Train Support Vector Machine (SVM)
11. Save trained model
12. Predict new email messages

---

## 📊 Model Used

* Support Vector Machine (SVM)

SVM was selected for deployment because it achieved the best performance among the models tested during experimentation.

Other models explored during development include:

* Bernoulli Naive Bayes
* Logistic Regression
* Random Forest

---

## 📸 Demo

Enter any email message into the Streamlit application.

Example:

```
Congratulations!

You have won a FREE iPhone.

Click the link below to claim your prize.
```

Prediction:

```
🚨 Spam Email
```

---

## 📚 Learning Outcomes

This project helped me understand:

* NLP preprocessing
* Text vectorization using TF-IDF
* Spam email classification
* Machine Learning model training
* Model serialization using Joblib
* Building and deploying a Streamlit application

---

## 🌐 Live Demo

https://email-spam-classifier-9ezwd7up4y97ed4jqdbwyx.streamlit.app/

---

## 👨‍💻 Author

**Himanshu Pal**

If you found this project helpful, feel free to ⭐ the repository.
