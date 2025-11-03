
## 📰 Fake News Detection using Machine Learning

### 📌 Overview

This project detects whether a given news article is **real** or **fake** using **Natural Language Processing (NLP)** and **Machine Learning**.
It uses text preprocessing, TF-IDF vectorization, and a trained **PassiveAggressiveClassifier / SVM** model to classify news as *True* or *Fake*.

---

### ⚙️ Features

✅ Clean dataset of real and fake news articles
✅ Preprocessing of text using **NLTK** (tokenization, stopword removal, etc.)
✅ **TF-IDF Vectorizer** for feature extraction
✅ **Machine Learning model** for classification
✅ **Streamlit App** for easy user interface
✅ Model saved and loaded using **Joblib**

---

### 🧠 Tech Stack

* **Python 3**
* **Pandas** — Data handling
* **NLTK** — Text preprocessing
* **Scikit-learn** — ML algorithms (TF-IDF, train-test split, PassiveAggressiveClassifier / SVM)
* **Streamlit** — Web app interface
* **Joblib** — Model saving/loading

---

### 📁 Project Structure

```
fake-news-detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── src/
│   ├── train_model.py        # Trains the model
│   ├── predict.py            # Predicts news authenticity
│
├── app.py                    # Streamlit web app
├── model.pkl                 # Saved ML model
├── README.md                 # Project documentation
└── requirements.txt          # All dependencies
```

---

### 🚀 How to Run the Project

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/Akash-kr-gupta/fake-news-detection.git
cd fake-news-detection
```

#### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Then open the URL shown (usually `http://localhost:8501`) in your browser.

---

### 📊 Dataset

* **Fake.csv** — Contains fake news articles
* **True.csv** — Contains genuine news articles
  (Source: Kaggle Fake News Dataset)

---

### 📈 Model Used

* **TF-IDF Vectorizer**: Converts text to numerical features
* **PassiveAggressiveClassifier / SVM**: Classifies news into *True* or *Fake*
* Accuracy achieved: ~93–96% (depending on dataset and split)

---

### 🧩 Example Output

| Input News Headline                          | Prediction |
| -------------------------------------------- | ---------- |
| "Government launches new scheme for farmers" | ✅ True     |
| "Aliens spotted in New York City park"       | ❌ Fake     |

---

### 💾 Save & Load Model

The trained model is saved using Joblib for fast loading:

```python
import joblib
model = joblib.load("model.pkl")
```

---

### 📌 Future Improvements

* Add deep learning models (LSTM / BERT)
* Expand dataset for better accuracy
* Integrate with a browser extension for real-time fake news detection

---

### 👨‍💻 Author

**Akash Kumar Gupta**
📧 ak01gupta8235@gmail.com
🌐 GitHub Profile (https://github.com/Akash-kr-gupta)

---

