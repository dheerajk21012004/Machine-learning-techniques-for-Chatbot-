# Machine-learning-techniques-for-Chatbot-
Machine learning techniques for chatbot data
# 🤖 Chatbot Response Classification — Machine Learning

A multi-model NLP classification pipeline built in Python to predict chatbot response categories for a real-world dataset provided by **Japeto**. Developed as coursework for **MOD006562 Machine Learning** at ARU (Anglia Ruskin University), 2024/25 Trimester 2.

The goal was to classify user–chatbot conversation pairs into **20 categories** and achieve ≥85% accuracy.

---

## 📁 Repository Structure

```
├── Machine_learning.ipynb      # Main Jupyter notebook (full pipeline)
├── chatbot_dataset.xlsx        # Input dataset (1358 rows, provided by Japeto)
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | Japeto (company-provided) |
| Format | Excel (.xlsx) |
| Original rows | 1,358 |
| After augmentation | 2,716 |
| Categories (classes) | 20 |
| Key columns used | `user_message`, `chatbot_response`, `categories` |
| Dropped columns | `session_id`, `message_time`, `intent_name`, `response_source` |

The dataset was significantly **class-imbalanced** across 20 categories.

---

## 🔧 Pipeline Overview

### 1. Exploratory Data Analysis
- Checked for null values and duplicates (none found)
- Visualised class distribution using Seaborn — confirmed heavy imbalance

### 2. Data Pre-processing

| Step | Method |
|------|--------|
| Grammar correction | LanguageTool API (Java-based, applied to `user_message`) |
| Data augmentation | NLTK WordNet synonym replacement on `user_message` — doubled dataset size |
| Lowercasing | Python `string` library on both text columns |
| Lemmatization | NLTK `WordNetLemmatizer` (preferred over stemming) |
| Stopword removal | NLTK `stopwords` corpus |

> Grammar correction and synonym augmentation each contributed a measurable increase in model accuracy.

### 3. Vectorization

- **TF-IDF** (`TfidfVectorizer`, `ngram_range=(1,1)`) applied independently to `user_message` and `chatbot_response`
- Vectors concatenated using `numpy.hstack` → final feature matrix shape: **(2716 × 3548)**

### 4. Train / Test Split & Validation

- **80/20 train-test split** (`sklearn train_test_split`, `random_state=10`)
- **10-fold cross-validation** (`KFold`, `cross_val_score`) used for all models

---

## 🧠 Models & Results

| Model | Test Accuracy | CV Score Range | Recommended? |
|-------|:------------:|:--------------:|:------------:|
| Naïve Bayes (Multinomial) | 73.89% | 68–75% | ⚠️ Acceptable |
| K-Nearest Neighbours (k=3) | 68.93% | 58–78% | ❌ Too low |
| Logistic Regression | 82.54% | ~83% | ✅ Yes |
| Random Forest (100 trees) | **90.62%** | 92–99% | ✅ **Best** |
| SVM (linear kernel) | 88.79% | 91–97% | ✅ Yes |

### Impact of Data Augmentation

| Model | Without Augmentation | With Augmentation | Gain |
|-------|:-------------------:|:-----------------:|:----:|
| Naïve Bayes | 58.45% | 73.89% | +15.44% |
| KNN | 55.51% | 68.93% | +13.42% |
| Logistic Regression | 70.22% | 82.54% | +12.32% |
| Random Forest | 68.01% | 90.62% | +22.61% |
| SVM | 71.23% | 88.79% | +17.56% |

> Synonym-based augmentation provided the largest single accuracy boost across all models, particularly for Random Forest.

---

## 🏆 Recommended Models

**Random Forest** is the top performer (90.62% accuracy, CV 92–99%) and clearly the best fit for this dataset. **SVM** (linear kernel) and **Logistic Regression** are strong alternatives with consistent validation scores close to their test accuracy, indicating good generalisation.

KNN and Naïve Bayes fall below the client's 85% target and are not recommended for production use.

---

## 🚀 Getting Started

### Requirements

```bash
pip install pandas numpy scikit-learn nltk openpyxl seaborn matplotlib requests
```

You will also need to download NLTK data on first run:

```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
```

### Running the Notebook

1. Place `chatbot_dataset.xlsx` in the same directory as the notebook
2. Open `Machine_learning.ipynb` in Jupyter or VS Code
3. Run all cells in order — the pipeline is fully sequential

> **Note:** The grammar correction step calls the [LanguageTool API](https://api.languagetool.org/v2/check). An internet connection is required for that cell, or you can skip it and use the dataset as-is.

---

## 📦 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Feature matrix concatenation |
| `scikit-learn` | Models, TF-IDF, train/test split, cross-validation, metrics |
| `nltk` | Synonym augmentation, lemmatization, stopword removal |
| `seaborn` / `matplotlib` | Visualisation (class distribution, confusion matrices) |
| `requests` | LanguageTool grammar correction API |

---

## ⚙️ Notes & Limitations

- Category 16 was consistently misclassified across all models — likely due to overlapping chatbot responses with other categories or mislabelling in the original dataset.
- Synonym augmentation doubles the dataset but preserves the existing class imbalance proportionally; a future improvement would be targeted oversampling (e.g. SMOTE) for underrepresented categories.
- The LanguageTool grammar correction step is slow on large datasets — consider running it once and saving the corrected dataset separately.

---
## 🔑 API Setup — Grammar Correction (LanguageTool)

One cell in the notebook uses the free LanguageTool API to auto-correct 
grammatical errors in `user_message`. No API key is required for the 
free tier — it works via a simple HTTP POST request.

The relevant cell hits this endpoint:
https://api.languagetool.org/v2/check

**Requirements:**
- An internet connection when running that cell
- The `requests` library (`pip install requests`)

**Free tier limits:** 20 requests/minute, 75KB max text per request.  
If you hit rate limits, add a `time.sleep(1)` inside the loop.

**To skip it entirely:** The cell is self-contained — you can skip it 
and the rest of the notebook will still run fine using the uncorrected text.


## 📄 References

- LanguageTool API — https://api.languagetool.org/v2/check
- NLTK — https://www.nltk.org
- scikit-learn — https://scikit-learn.org
- OpenAI ChatGPT (used for debugging assistance only)
