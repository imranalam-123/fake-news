print("TRAINING STARTED...")

import pandas as pd
import re
import pickle
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")

# =========================
# LOAD KAGGLE DATA
# =========================
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title + text (VERY IMPORTANT)
df["text"] = df["title"].astype(str) + " " + df["text"].astype(str)

print("Label distribution:")
print(df["label"].value_counts())

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# MODEL
# =========================
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("\nMODEL SAVED: model/fake_news_model.pkl")
