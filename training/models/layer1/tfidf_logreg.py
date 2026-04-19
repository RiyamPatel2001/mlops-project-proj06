"""
models/tfidf_logreg.py
──────────────────────
TF-IDF + Logistic Regression baseline for Layer-1 transaction categorization.

This is Candidate 1 in the MLflow runs table — the floor every other model
must beat.  All hyperparameters come from config so MLflow can track them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train(
    X_train: pd.Series | list[str],
    y_train: np.ndarray,
    config: dict,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """
    Fit a TF-IDF vectorizer and a Logistic Regression classifier.

    Args:
        X_train: payee_norm strings (normalized via preprocess.normalize_payee)
        y_train: integer-encoded category labels
        config:  dict — reads from the tfidf_logreg section of config.yaml:
                   config["tfidf_logreg"]["C"]             (float, default 1.0)
                   config["tfidf_logreg"]["class_weight"]  (str,   default "balanced")
                   config["tfidf_logreg"]["max_iter"]      (int,   default 1000)

    Returns:
        (vectorizer, classifier) — both fitted, ready for evaluate_and_log()
    """
    model_cfg = config.get("tfidf_logreg", {})

    # ── Vectorizer ────────────────────────────────────────────────────────────
    # char_wb: character n-grams with word-boundary padding.
    # Captures sub-word patterns (e.g. "STAR" in STARBUCKS, "BUCKS")
    # without needing a vocabulary of full merchant names.
    # ngram_range=(2,4): bi- to 4-grams; covers abbreviations and suffixes.
    # sublinear_tf=True: replaces raw TF with 1+log(TF) — dampens very
    # frequent n-grams so rare but discriminative ones get fair weight.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=50_000,
        sublinear_tf=True,
    )

    # ── Classifier ───────────────────────────────────────────────────────────
    classifier = LogisticRegression(
        max_iter=model_cfg.get("max_iter", 1000),
        class_weight=model_cfg.get("class_weight", "balanced"),
        C=model_cfg.get("C", 1.0),
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    X_vec = vectorizer.fit_transform(X_train)
    classifier.fit(X_vec, y_train)

    return vectorizer, classifier