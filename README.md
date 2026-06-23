# 📧 Spam Detection System

A complete end-to-end **SMS Spam Detection** system built with Python and Scikit-learn. The project covers the full ML pipeline — from raw text preprocessing to model training, hyperparameter tuning, evaluation, and a ready-to-use prediction function.

---

## 📌 Project Overview

| Item | Details |
|---|---|
| **Task** | Binary Text Classification (Ham vs Spam) |
| **Dataset** | UCI SMS Spam Collection (~5,574 messages) |
| **Vectorizer** | TF-IDF with bigrams |
| **Models** | Naive Bayes, Logistic Regression, LinearSVC, Random Forest, Voting Ensemble |
| **Best Model** | LinearSVC (Tuned) — ~98–99% Accuracy |
| **Language** | Python 3.10+ |

---

## 📁 Repository Structure

```
spam-detection-system/
│
├── spam_detection.ipynb        # Main Jupyter Notebook (complete pipeline)
├── requirements.txt            # Python dependencies
├── spam_detector_model.pkl     # Saved trained model (generated on run)
│
├── outputs/                    # Charts saved during notebook execution
│   ├── eda_distribution.png
│   ├── wordclouds.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
│
└── README.md
```

---

## 📊 Dataset

**UCI SMS Spam Collection Dataset**

- 5,574 real SMS messages labeled as `ham` (legitimate) or `spam`
- Widely used NLP benchmark for binary text classification
- Loaded automatically in the notebook — no manual download needed

| Source | Link |
|---|---|
| Primary (GitHub mirror) | https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv |
| Official UCI Repository | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |

---

## 🔄 Pipeline Walkthrough

### 1. 📦 Installation & Imports
All required libraries are installed and imported. NLTK corpora (stopwords, punkt) are auto-downloaded.

### 2. 📂 Data Loading
Dataset is fetched directly from a public URL. A fallback to the UCI ZIP archive is included.

### 3. 🔍 Exploratory Data Analysis (EDA)
- Class distribution (Ham vs Spam counts)
- Message length statistics by class
- Word count, digit count, uppercase letter analysis
- Distribution plots and box plots
- Word clouds for ham and spam messages

### 4. 🧹 Text Preprocessing
Each message goes through a full NLP cleaning pipeline:
- Lowercase conversion
- URL, email, and phone number replacement with tokens
- Punctuation and special character removal
- Number normalization
- Stopword removal (NLTK English stopwords)
- Porter Stemming

### 5. ✂️ Feature Engineering & Train/Test Split
- Label encoding: `ham → 0`, `spam → 1`
- Duplicate removal
- 80/20 stratified train/test split

### 6. 🤖 Model Training
Five Scikit-learn Pipelines trained (TF-IDF + Classifier):

| Model | Notes |
|---|---|
| Multinomial Naive Bayes | Classic NLP baseline |
| Logistic Regression | Strong linear model |
| LinearSVC | SVM — best for high-dimensional text |
| Random Forest | Ensemble tree model |
| Voting Ensemble (Soft) | Combines NB + LR + RF |

Each model is evaluated on: Test Accuracy, F1-Score, AUC-ROC, and 5-Fold CV Accuracy.

### 7. 📊 Model Comparison
Grouped bar chart and horizontal accuracy chart comparing all 5 models across all metrics.

### 8. 🏆 Best Model Evaluation
Deep-dive evaluation of the best model:
- Classification Report (Precision, Recall, F1 per class)
- Confusion Matrix heatmap
- ROC Curve with AUC score
- Precision-Recall Curve

### 9. 🔎 Feature Importance
Top 20 spam and ham indicator words visualized using Logistic Regression coefficients.

### 10. 🔧 Hyperparameter Tuning
`GridSearchCV` (5-fold CV) on LinearSVC pipeline, tuning:
- `max_features`: [8000, 15000]
- `ngram_range`: [(1,1), (1,2)]
- `min_df`: [1, 2]
- `C`: [0.5, 1.0, 5.0]

### 11. 🚀 Final Model & Prediction
- Best tuned model saved as `spam_detector_model.pkl`
- `predict_spam()` function for real-world use
- Tested on 7 real-world messages

### 12. 📈 Results Summary
Final performance table and classification report printed.

---

## 📈 Results

| Model | Test Accuracy | F1-Score |
|---|---|---|
| LinearSVC (Tuned) | ~98–99% | ~0.99 |
| Voting Ensemble | ~98% | ~0.98 |
| Logistic Regression | ~97–98% | ~0.98 |
| Naive Bayes | ~97% | ~0.97 |
| Random Forest | ~97% | ~0.97 |

> **Why LinearSVC?** Text data is high-dimensional and sparse. SVMs find the optimal separating hyperplane in such spaces, making LinearSVC both fast and highly accurate for text classification tasks.

---

## ⚙️ Setup and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/RabiyaMalik242/spam-detection-system.git
cd spam-detection-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook spam_detection.ipynb
```

Run all cells top to bottom. The dataset loads automatically.

### 4. Use the Prediction Function
After running the notebook, use the saved model directly:

```python
import pickle

with open('spam_detector_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_spam(message, model):
    import re, string
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = message.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'url', text)
    text = re.sub(r'\S+@\S+', 'email', text)
    text = re.sub(r'\b\d{10,}\b|\+\d{1,3}\s?\d+', 'phone', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b\d+\b', 'num', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    cleaned = ' '.join(tokens)

    pred = model.predict([cleaned])[0]
    return "🚨 SPAM" if pred == 1 else "✅ HAM"

print(predict_spam("Congratulations! You won a FREE prize. Call now!", model))
# Output: 🚨 SPAM

print(predict_spam("Hey, are we meeting at 5pm today?", model))
# Output: ✅ HAM
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, pipelines, evaluation |
| `nltk` | Text preprocessing (stopwords, stemming, tokenization) |
| `matplotlib` | Plotting and visualizations |
| `seaborn` | Statistical visualizations (heatmaps) |
| `wordcloud` | Word cloud generation |
| `pickle` | Model serialization |

---

## 🙋‍♀️ Author

**Rabiya**
BS Software Engineering

---

