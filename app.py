import streamlit as st
import pickle
import os

# -----------------------------
# Page config (must be first)
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Load model & vectorizer safely
# -----------------------------
MODEL_PATH = "model/fake_news_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please train the model and push it to GitHub.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model, vectorizer = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detection")
st.write("Enter a news article below to check whether it is **REAL** or **FAKE**.")

text = st.text_area("News Text", height=220)

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -----------------------------
# Predict
# -----------------------------
if st.button("Check News"):
    if not text.strip():
        st.warning("⚠️ Please enter some news text")
    else:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        st.session_state.prediction = prediction

        if prediction == "FAKE":
            st.error("🚨 This news is **FAKE**")
        else:
            st.success("✅ This news is **REAL**")

# -----------------------------
# Raw output
# -----------------------------
st.markdown("---")
st.subheader("🔎 Raw Prediction Output")

if st.session_state.prediction is not None:
    st.code({"prediction": st.session_state.prediction})
