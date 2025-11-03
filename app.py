import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy as np

# --- 1. CONFIGURATION AND MODEL/DATA LOADING ---

MODEL_PATH = 'models/fake_news_pipeline.pkl'

@st.cache_resource
def load_model():
    """Loads the trained model pipeline."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"❌ Model file not found at '{MODEL_PATH}'. Please train and save your model first.")
        return None

@st.cache_resource
def download_nltk_data():
    """Downloads NLTK data if missing."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

# Initialize
download_nltk_data()
model = load_model()

# --- 2. TEXT CLEANING FUNCTION ---

def clean_text(text):
    """Cleans the input text."""
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# --- 3. PREDICTION FUNCTION ---

def predict_news(text, model):
    """Predicts whether the news is real or fake."""
    if not model:
        return "MODEL ERROR", 0.0

    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    decision_value = model.decision_function([cleaned])[0]
    confidence = 1 / (1 + np.exp(-decision_value))
    label = "REAL" if prediction in ["REAL", 1] else "FAKE"
    return label, confidence * 100

# --- 4. STREAMLIT APP UI ---

def main():
    st.set_page_config(page_title="📰 Fake News Detector", page_icon="🤖", layout="centered")
    st.title("📰 Fake News Detection App")
    st.markdown("Enter any news article or headline below to predict if it's real or fake.")

    user_input = st.text_area("🗞️ Enter News Text:", height=150, placeholder="The government announced a new education policy...")

    if st.button("Check Authenticity", use_container_width=True, type="primary"):
        if user_input.strip():
            with st.spinner('Analyzing news text...'):
                label, confidence = predict_news(user_input, model)

            st.markdown("---")

            if label == "REAL":
                st.success(f"✅ **Prediction:** REAL")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                st.balloons()
            else:
                st.error(f"🚫 **Prediction:** FAKE")
                st.markdown(f"**Confidence:** {confidence:.2f}%")

            st.info("ℹ️ Model used: PassiveAggressiveClassifier (TF-IDF + NLP Pipeline)")
        else:
            st.warning("⚠️ Please enter some text first.")

if __name__ == "__main__":
    main()
