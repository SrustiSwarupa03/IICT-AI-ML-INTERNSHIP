import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from pathlib import Path

# Check/download English stopwords
nltk.download("stopwords", quiet=True)

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load trained model and TF-IDF vectorizer
model = joblib.load(BASE_DIR / "phishing_email_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")


# Same preprocessing used during model training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    clean_words = [word for word in words if word not in stop_words]
    return " ".join(clean_words)


# Page configuration
st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="📧",
    layout="centered"
)


# Header
st.title("📧 AI Phishing Email Detector")

st.markdown(
    """
    ### Protect yourself from suspicious emails

    This AI-powered system analyzes email text using **Natural Language Processing (NLP)**
    and a trained **Neural Network** to classify an email as:

    - ✅ **Safe Email**
    - 🚨 **Phishing Email**
    """
)

st.divider()


# Email input
st.subheader("📨 Enter Email Content")

email_text = st.text_area(
    "Paste the email you want to analyze:",
    height=250,
    placeholder="Paste the complete email content here..."
)


# Analyze button
if st.button("🔍 Analyze Email", type="primary", use_container_width=True):

    if not email_text.strip():

        st.warning("⚠️ Please enter an email before analyzing.")

    else:

        # Clean email
        cleaned_email = clean_text(email_text)

        # Convert text into TF-IDF features
        email_vector = vectorizer.transform([cleaned_email])

        # Prediction
        prediction = int(model.predict(email_vector)[0])

        st.divider()
        st.subheader("📊 Analysis Result")

        # Display result
        if prediction == 0:

            st.error(
                "🚨 PHISHING EMAIL\n\n"
                "This email has been classified as potentially malicious."
            )

        else:

            st.success(
                "✅ SAFE EMAIL\n\n"
                "This email has been classified as safe by the model."
            )

        # Confidence
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(email_vector)[0]
            confidence = max(probabilities) * 100

            st.metric(
                label="🤖 Model Confidence",
                value=f"{confidence:.2f}%"
            )


st.divider()

st.caption(
    "⚠️ This AI prediction is not a guarantee. "
    "Always verify suspicious emails through trusted channels."
)

st.caption(
    "AI Powered Phishing Email Detection • IICT Internship Project"
)