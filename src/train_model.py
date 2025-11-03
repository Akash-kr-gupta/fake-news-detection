# src/train_model.py

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
import joblib

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# ✅ Load datasets
fake = pd.read_csv('data/fake.csv')
true = pd.read_csv('data/true.csv')

# ✅ Add labels
fake['label'] = 'FAKE'
true['label'] = 'REAL'

# ✅ Combine both datasets
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ✅ Prepare features and labels
X = data['text']
y = data['label']

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Build model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('model', PassiveAggressiveClassifier(max_iter=50))
])

# ✅ Train model
pipeline.fit(X_train, y_train)

# ✅ Evaluate model
score = pipeline.score(X_test, y_test)
print(f"\n📊 Model Accuracy: {score * 100:.2f}%")

# ✅ Save trained model
joblib.dump(pipeline, 'models/fake_news_pipeline.pkl')
print("✅ Model saved to 'models/fake_news_pipeline.pkl'")


 #To check the accuracy run:-     python src/train_model.py
