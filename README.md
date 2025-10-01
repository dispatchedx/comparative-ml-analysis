# Predictive Modeling in Python

This repository contains a collection of data mining and machine learning projects implemented in Python, showcasing data preprocessing, feature engineering, and model evaluation.

---

## Projects Included

### 1. Wine Quality Prediction

- **Goal:** To predict the quality of red wine based on its chemical properties.
- **Dataset:** [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) from UCI Machine Learning Repository..
- **Techniques Used:**
  - Data cleaning and analysis with **Pandas**.
  - Model training using **Scikit-learn's** Support Vector Classifier (SVC).
  - Data imputation strategies for missing pH values, including replacement with the mean, prediction via Linear Regression, and K-Means clustering averages.

### 2. "The Onion" News Headline Classifier

- **Goal:** To classify news headlines as either real news or satire from "The Onion."
- **Dataset:** ["Onion or Not" dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) from Kaggle.
- **Techniques Used:**
  - Text preprocessing pipeline using **NLTK** for tokenization, stemming, stop-word removal, and punctuation cleaning.
  - Feature engineering with text vectorization using **Keras Tokenizer**.
  - Built and trained a Sequential neural network with **Keras** using Embedding and Dense layers for binary classification.
