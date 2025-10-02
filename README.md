# Data Mining & Machine Learning Projects

This repository contains a collection of Python scripts demonstrating various data mining and machine learning techniques. The projects focus on data preprocessing, feature engineering, model training, and evaluation for both tabular and text-based data.

---

## Projects Overview

### 1. News Headline Classifier ("The Onion" vs. Real News)

This project tackles a binary text classification problem: determining whether a news headline is from the satirical publication "The Onion" or a legitimate news source.

**Key Features:**

- **Text Preprocessing:** Implements a complete text preprocessing pipeline using `nltk`, including lowercasing, punctuation removal, stop-word removal, and stemming.
- **Vectorization:** Prepares text data for modeling using Tokenization and Padding with `tensorflow.keras`.
- **Model Comparison:** The `news_headline_classifier_multimodel.py` script systematically trains, evaluates, and compares a wide array of deep learning and machine learning architectures for text classification:
  - A simple Dense Neural Network with an Embedding layer.
  - Recurrent Neural Networks: **LSTM** and **Bidirectional LSTM** models to capture sequential context.
  - **Convolutional Neural Network (CNN)** for detecting local patterns (n-grams) in text.
  - A model based on **TF-IDF** features.
  - **Sentence Transformers (BERT-family models):** Leverages powerful pre-trained models like `all-MiniLM-L6-v2` for generating high-quality sentence embeddings, which are then fed into a Logistic Regression classifier.

## Model Performance Comparison

### Traditional Neural Networks (with stemming & stopword removal)

| Model                      | Accuracy   | Notes                                            |
| -------------------------- | ---------- | ------------------------------------------------ |
| **Bidirectional LSTM**     | **84.95%** | Best performer - reads text forwards & backwards |
| Deep Dense Model           | 84.13%     | Multiple dense layers with dropout               |
| CNN                        | 84.07%     | Detects local n-gram patterns (like "area man")  |
| LSTM                       | 83.85%     | Sequential context modeling                      |
| TF-IDF + Dense NN          | 83.52%     | Bag-of-words with neural network                 |
| Simple Embedding + Pooling | 83.47%     | Baseline embedding model                         |

### BERT-family models Sentence Transformers (no preprocessing)

| Model                   | Accuracy | Notes                                 |
| ----------------------- | -------- | ------------------------------------- |
| TF-IDF (comparison)     | 83.33%   | Traditional approach outperforms BERT |
| paraphrase-MiniLM-L3-v2 | 83.08%   | Smallest/fastest (17M params)         |
| all-MiniLM-L6-v2        | 81.83%   | Medium model (22M params)             |

Note: Sentence Transformer models were evaluated without preprocessing, since they are pre-trained on raw natural text.

### Key Findings

- **Bidirectional LSTM achieved best results** at 84.95% accuracy
- **BERT-family models Sentence Transformers** underperformed (81–83%) compared to simpler neural models. Likely reasons: short inputs, small dataset size, and the satirical “style” being more about lexical cues than deep semantic meaning.
- **Sweet spot:** Medium-complexity models (LSTM/CNN) that capture sequential patterns without over-engineering
- **Dataset:** 24,000 headlines (75% train, 25% test) - Onion vs. real news classification

### 2. Wine Quality Predictor

This project aims to predict the quality of red wine based on its chemical properties using the "Wine Quality" dataset.

**Key Features:**

- **Model Training:** Implements and evaluates several classification models, including Support Vector Machines (SVC) and Random Forests, to predict wine quality scores.
- **Handling Missing Data:** A significant focus of this project is exploring different strategies for data imputation. When 33% of values in the 'alcohol' column are intentionally removed, the script evaluates the impact of four different recovery methods:
  1.  **Removing the Column:** Dropping the feature entirely.
  2.  **Mean Imputation:** Filling missing values with the column's average.
  3.  **Regression Imputation:** Training a Linear Regression model on other features to predict and fill the missing alcohol values.
  4.  **K-Means Cluster Imputation:** Grouping data into clusters and filling missing values with the average of the specific cluster they belong to.

---

## Technologies Used

- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Deep Learning:** tensorflow (Keras)
- **NLP:** nltk, sentence-transformers
