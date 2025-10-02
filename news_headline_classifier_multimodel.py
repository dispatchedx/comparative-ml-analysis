import pandas as pd

import re

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, metrics

from sklearn.linear_model import LogisticRegression

from tensorflow.keras import Sequential

from tensorflow.keras.layers import (
    Dense,
    Embedding,
    GlobalAveragePooling1D,
    Dropout,
    LSTM,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
)

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentence_transformers import SentenceTransformer

import numpy as np


nltk.download("punkt")

nltk.download("stopwords")


# Read csv

data = pd.read_csv("data/OnionOrNot.csv")


# =============================================================================

# PREPROCESSING FUNCTIONS

# =============================================================================


def preprocess_for_embedding(dataframe):
    """Preprocessing for embedding-based models"""

    def clean_text(text):

        text = text.lower()

        text = re.sub(r"[!@#$%&'()*+,\\'-./:;" '<=>?"\[\]^_—`{|}~\d…]', " ", text)

        text = " ".join(text.split())

        return text

    def remove_stopwords_and_stem(text):

        stop_words = set(nltk.corpus.stopwords.words("english"))

        stemmer = nltk.stem.PorterStemmer()

        words = text.split()

        words = [word for word in words if word not in stop_words]

        words = [stemmer.stem(word) for word in words]

        return " ".join(words)

    texts = dataframe["text"].apply(clean_text)

    texts = texts.apply(remove_stopwords_and_stem)

    tokenizer = Tokenizer(num_words=8000)

    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    X = pad_sequences(sequences, maxlen=100)

    return X, texts.tolist()


def preprocess_for_tfidf(dataframe):
    """Preprocessing for TF-IDF model"""

    def clean_text(text):

        text = text.lower()

        text = re.sub(r"[!@#$%&'()*+,\\'-./:;" '<=>?"\[\]^_—`{|}~\d…]', " ", text)

        text = " ".join(text.split())

        return text

    def remove_stopwords_and_stem(text):

        stop_words = set(nltk.corpus.stopwords.words("english"))

        stemmer = nltk.stem.PorterStemmer()

        words = text.split()

        words = [word for word in words if word not in stop_words]

        words = [stemmer.stem(word) for word in words]

        return " ".join(words)

    texts = dataframe["text"].apply(clean_text)

    texts = texts.apply(remove_stopwords_and_stem)

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(texts).toarray()

    return X


# =============================================================================

# MODEL ARCHITECTURES

# =============================================================================


def build_simple_embedding_model(vocab_size=8000):
    """Original simple model"""

    model = Sequential(
        [
            Embedding(vocab_size + 1, 128),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def build_lstm_model(vocab_size=8000):
    """LSTM model - captures sequential patterns and context"""

    model = Sequential(
        [
            Embedding(vocab_size + 1, 128),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # LSTM layer with dropout
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def build_bilstm_model(vocab_size=8000):
    """Bidirectional LSTM - reads text forwards AND backwards"""

    model = Sequential(
        [
            Embedding(vocab_size + 1, 128),
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def build_cnn_model(vocab_size=8000):
    """CNN model - detects local patterns (like n-grams)"""

    model = Sequential(
        [
            Embedding(vocab_size + 1, 128),
            Conv1D(128, 5, activation="relu"),  # Detect 5-word patterns
            MaxPooling1D(pool_size=4),
            Conv1D(64, 3, activation="relu"),  # Detect 3-word patterns
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def build_deep_model(vocab_size=8000):
    """Deeper network - more layers to learn complex patterns"""

    model = Sequential(
        [
            Embedding(vocab_size + 1, 128),
            GlobalAveragePooling1D(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def build_tfidf_model(input_dim):
    """TF-IDF Dense model"""

    model = Sequential(
        [
            Dense(128, activation="relu", input_dim=input_dim),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


# =============================================================================

# TRAINING FUNCTION

# =============================================================================


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Train model and print results"""

    print(f"\n{'='*70}")

    print(f"{model_name}")

    print("=" * 70)

    model.summary()

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[es], verbose=1
    )

    predictions = model.predict(X_test, verbose=0)

    y_pred = (predictions > 0.5).astype(int).flatten()

    print(f"\n{model_name} Results:")

    print(metrics.classification_report(y_test, y_pred))

    return history, y_pred


# =============================================================================

# SENTENCE TRANSFORMER MODELS (BERT-based)

# =============================================================================


def preprocess_for_bert(dataframe):
    """Minimal preprocessing for BERT - keep natural language intact!"""

    # Just nothing really. - NO stemming, NO stopword removal

    # texts = dataframe["text"].str.lower().str.strip()
    texts = dataframe["text"].str.strip()
    return texts.tolist()


def test_sentence_transformers(dataframe, target):
    """Test multiple pre-trained language models"""

    # Get minimally processed text (BERT likes natural language!)

    texts = preprocess_for_bert(dataframe)

    # Three popular sentence transformer models

    models_to_test = [
        ("all-MiniLM-L6-v2", "Fast & Efficient (22M params)"),
        ("paraphrase-MiniLM-L3-v2", "Smallest & Fastest (17M params)"),
        # ("all-mpnet-base-v2", "Best Quality (110M params) - VERY SLOW"),  # Uncomment if you want best quality (still minor difference)
    ]

    bert_results = {}

    for model_name, description in models_to_test:

        print(f"\n{'='*70}")

        print(f"SENTENCE TRANSFORMER: {model_name}")

        print(f"Description: {description}")

        print("=" * 70)

        # Load pre-trained model

        print(f"Loading {model_name}...")

        encoder = SentenceTransformer(model_name)

        # Encode texts (this takes a while)

        print("Encoding texts...")

        X = encoder.encode(texts, show_progress_bar=True, batch_size=64)

        # Split data

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, target, test_size=0.25, random_state=1)

        # Simple logistic regression on top

        print("Training classifier...")

        clf = LogisticRegression(max_iter=1000, random_state=1)

        clf.fit(X_train, y_train)

        # Predict

        y_pred = clf.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)

        bert_results[f"BERT: {model_name}"] = accuracy

        print(f"\n{model_name} Results:")

        print(metrics.classification_report(y_test, y_pred))

    return bert_results


# =============================================================================

# MAIN EXECUTION

# =============================================================================


target = data["label"].values


# Get preprocessed data

X_emb, texts = preprocess_for_embedding(data)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_emb, target, test_size=0.25, random_state=1)


# Test all neural network models

models = [
    (build_simple_embedding_model(), "SIMPLE EMBEDDING MODEL (Original)"),
    (build_lstm_model(), "LSTM MODEL (Sequential Context)"),
    (build_bilstm_model(), "BIDIRECTIONAL LSTM MODEL (Both Directions)"),
    (build_cnn_model(), "CNN MODEL (Local Patterns)"),
    (build_deep_model(), "DEEP DENSE MODEL (More Layers)"),
]


results = {}

choice = input("Train neural network models? y/n\n")
if choice == "y":
    for model, name in models:
        history, y_pred = train_and_evaluate(model, X_train, X_test, y_train, y_test, name)
        results[name] = metrics.accuracy_score(y_test, y_pred)


# TF-IDF Model

print(f"\n{'='*70}")

print("TF-IDF MODEL")

print("=" * 70)

X_tfidf = preprocess_for_tfidf(data)

X_train_tfidf, X_test_tfidf, y_train, y_test = model_selection.train_test_split(
    X_tfidf, target, test_size=0.25, random_state=1
)

model_tfidf = build_tfidf_model(input_dim=X_tfidf.shape[1])

history_tfidf, y_pred_tfidf = train_and_evaluate(model_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF")

results["TF-IDF"] = metrics.accuracy_score(y_test, y_pred_tfidf)


# Sentence Transformers (BERT-based models)

print("\n" + "=" * 70)

print("TESTING SENTENCE TRANSFORMERS (This will take a few minutes...)")

print("=" * 70)

bert_results = test_sentence_transformers(data, target)

results.update(bert_results)


# Final summary

print("\n" + "=" * 70)

print("FINAL ACCURACY SUMMARY")

print("=" * 70)

for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):

    print(f"{name:45s}: {acc:.4f}")
