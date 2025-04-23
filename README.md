# 📰 Fake News Detection using NLP and Logistic Regression

A machine learning project that classifies news as **Real** or **Fake** using Natural Language Processing (NLP) and Logistic Regression.

---

## 📂 Dataset

- **Source**: [Kaggle - Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- **Files Used**: 
  - `True.csv`: Real news articles
  - `Fake.csv`: Fake news articles

---

## 🎯 Objective

Build a classification model using NLP techniques to differentiate between real and fake news articles.

---

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- SpaCy
- TfidfVectorizer
- Matplotlib / Seaborn (for visualization)

---

## 🧹 Text Preprocessing Steps

1. Convert all text to lowercase
2. Remove non-alphabetic characters
3. Tokenize the text
4. Remove stopwords
5. Apply stemming and lemmatization
6. Reconstruct the cleaned sentence

---

## 🧪 Model Building

- Used `TfidfVectorizer` to convert preprocessed text to numerical features.
- Applied `train_test_split` to divide the dataset into training and testing sets.
- Trained a **Logistic Regression** model on the training data.
- Evaluated the model on the test set using standard classification metrics.

---

## 📊 Evaluation Metrics

- **Accuracy**: 0.9857  
- **Precision**: 0.9837  
- **Recall**: 0.9863  
- **F1 Score**: 0.9850

```
Classification Report:

               precision    recall  f1-score   support

           0       0.99      0.99      0.99      4633
           1       0.98      0.99      0.98      4221

    accuracy                           0.99      8854
   macro avg       0.99      0.99      0.99      8854
weighted avg       0.99      0.99      0.99      8854
```

---

## 🔍 Discussion

### ✅ Model Performance
- Achieved over **98% accuracy** on unseen data.
- The model balances **high precision and high recall**, minimizing both false positives and false negatives.

### ✅ Preprocessing Impact
- Proper text normalization and feature extraction using TF-IDF played a critical role in model accuracy.

### ✅ Algorithm Choice
- Logistic Regression provided fast, accurate, and interpretable results for this binary classification task.

---

## 📝 Conclusion

- **Strengths**: Efficient and accurate pipeline using classic ML techniques and well-crafted preprocessing.
- **Weaknesses**: Might miss deeper contextual cues—could be improved with transformer-based models.
- **Future Work**:
  - Try deep learning models (e.g., LSTM, BERT)
  - Deploy as a web API or Flask app
  - Explore explainability tools like LIME/SHAP

---

## 📁 Project Structure

```
├── True.csv
├── Fake.csv
├── model_training.ipynb
├── requirements.txt
└── README.md
```

---

## 📌 Author

- Harshad Jadhav  
- [LinkedIn](https://www.linkedin.com/in/harshad-jadhav-073a41256/)

