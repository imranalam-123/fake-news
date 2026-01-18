import streamlit as st
import pickle

# -----------------------------
# Load model & vectorizer
# -----------------------------
with open("model/fake_news_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detection")
st.write("Enter a news article below to check whether it is **REAL** or **FAKE**.")

# Text input (replaces HTML form)
text = st.text_area("News Text", height=220)

# Predict button (replaces POST route)
if st.button("Check News"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some news text")
    else:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]

        if prediction == "FAKE":
            st.error("🚨 This news is **FAKE**")
        else:
            st.success("✅ This news is **REAL**")

# -----------------------------
# Optional API-like output
# -----------------------------
st.markdown("---")
st.subheader("🔎 Raw Prediction Output")
if text.strip():
    st.code({"prediction": prediction if "prediction" in locals() else None})
