import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy as np
import sys

# --- NLTK Data Download ---
# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    # Use sys.stdout.write for quiet download status
    sys.stdout.write("Downloading NLTK data (punkt, stopwords)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    sys.stdout.write("Done.\n")

# --- Model Loading ---
try:
    # Load trained model (Ensure 'models/fake_news_pipeline.pkl' exists!)
    model = joblib.load('models/fake_news_pipeline.pkl')
except FileNotFoundError:
    print("Error: Model file 'models/fake_news_pipeline.pkl' not found.")
    print("Please ensure you have trained the model and saved it to the correct path.")
    sys.exit(1)


# --- Text Preprocessing Function ---
def clean_text(text):
    """
    Cleans the input text by removing non-alphabetic characters, 
    converting to lowercase, tokenizing, and removing stopwords.
    """
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove English stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)


# --- Prediction Function ---
def predict_news(text):
    """
    Predicts the likelihood of the news text being REAL or FAKE 
    and calculates a normalized confidence score.
    """
    cleaned = clean_text(text)
    
    # Get the prediction (e.g., "REAL" or "FAKE", or 1 or 0)
    prediction = model.predict([cleaned])[0]
    
    # Get the raw decision function value
    # PassiveAggressiveClassifier is typically a binary classifier, 
    # so we take the first (and usually only) value.
    decision_value = model.decision_function([cleaned])[0]
    
    # Convert decision_function value into a 0–1 confidence using logistic mapping (sigmoid)
    # This scales the raw margin into a probability-like score.
    confidence = 1 / (1 + np.exp(-decision_value))
    
    # Determine the output label string
    label = "REAL" if prediction in ["REAL", 1] else "FAKE"
    
    # Print the results
    print(f"\n📰 News: {text[:80]}...")
    print(f"🔍 Prediction: {label}")
    # Print confidence as a formatted percentage
    print(f"📈 Confidence Score: {confidence * 100:.2f}%")


# --- Main Execution Block ---
if __name__ == "__main__":
    text = input("Enter news text to check: ")
    if text:
        predict_news(text)
    else:
        print("No text entered. Exiting.")