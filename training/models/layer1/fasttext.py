"""
models/fasttext_model.py
────────────────────────
fastText candidate for Layer-1 transaction categorization.

fastText trains on a temporary file in its own format:
    __label__Groceries TESCO SUPERSTORE
    __label__Dining_Out DOMINOS

It returns a fastText model object wrapped in a thin sklearn-compatible
adapter so evaluate_and_log() can call .predict() on it uniformly.

Config keys read from config["fasttext"]:
    lr          — learning rate         (default 0.1)
    epoch       — number of epochs      (default 25)
    wordNgrams  — word n-gram range     (default 2)
    dim         — embedding dimension   (default 100)
"""

from __future__ import annotations

import os
import tempfile

import fasttext
import numpy as np
import pandas as pd

# fasttext-wheel 0.9.2 uses np.array(..., copy=False) which breaks on NumPy 2.x.
# requirements.txt pins numpy==1.26.* to avoid this. Guard here so the error
# is obvious if the pin is ever changed.
_np_major = int(np.__version__.split(".")[0])
if _np_major >= 2:
    raise ImportError(
        f"fasttext-wheel 0.9.2 is incompatible with NumPy {np.__version__}. "
        f"requirements.txt pins numpy==1.26.* — check your environment."
    )


class FastTextClassifier:
    """
    Thin sklearn-compatible wrapper around a fastText model.
    Exposes .predict() so evaluate_and_log() works without modification.
    """

    def __init__(self, model, label_classes: list[str]):
        self._model = model
        # fastText returns labels as "__label__Groceries" — build a lookup
        # from that format back to integer indices matching LabelEncoder order
        self._label_to_idx = {
            f"__label__{cls.replace(' ', '_')}": i
            for i, cls in enumerate(label_classes)
        }
        self._label_classes = label_classes

    def predict(self, X) -> np.ndarray:
        """
        Args:
            X: list of normalized payee strings
        Returns:
            integer label array aligned with LabelEncoder indices
        """
        if isinstance(X, (pd.Series, np.ndarray)):
            X = X.tolist()

        preds = []
        for text in X:
            # fastText returns ([label], [prob]) — we want top-1 label
            labels, _ = self._model.predict(text, k=1)
            label = labels[0]
            idx = self._label_to_idx.get(label, 0)
            preds.append(idx)
        return np.array(preds)


def _to_fasttext_format(X: pd.Series, y: np.ndarray, label_classes: list[str]) -> str:
    """
    Write a temp file in fastText's training format and return the path.
    fastText requires labels as __label__<name> with spaces replaced by _.
    Caller is responsible for deleting the file.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    for text, label_idx in zip(X, y):
        label_str = label_classes[label_idx].replace(" ", "_")
        tmp.write(f"__label__{label_str} {text}\n")
    tmp.close()
    return tmp.name


def train(
    X_train: pd.Series | list[str],
    y_train: np.ndarray,
    config: dict,
) -> tuple[None, FastTextClassifier]:
    """
    Train a fastText supervised classifier.

    Args:
        X_train: payee_norm strings
        y_train: integer-encoded category labels
        config:  full config dict — reads config["fasttext"] section

    Returns:
        (None, FastTextClassifier) — None in the vectorizer slot because
        fastText handles its own text processing internally; evaluate_and_log()
        already handles vec=None correctly.
    """
    model_cfg = config["fasttext"]

    # fasttext SGD diverges (NaN) at lr >= ~1.2 on typical transaction datasets
    if model_cfg["lr"] > 1.1:
        raise ValueError(
            f"fasttext lr={model_cfg['lr']} exceeds safe ceiling of 1.1 — "
            f"values above ~1.2 cause gradient explosion (RuntimeError: Encountered NaN). "
            f"Use lr <= 1.0."
        )

    # Reconstruct label_classes from the processed dir so the label→idx
    # mapping is consistent with what LabelEncoder produced in preprocess.py
    import json
    processed_dir = config["data"]["processed_dir"]
    with open(os.path.join(processed_dir, "label_classes.json")) as f:
        label_classes = json.load(f)

    # Write training data to a temp file in fastText format
    train_file = _to_fasttext_format(X_train, y_train, label_classes)

    try:
        model = fasttext.train_supervised(
            input=train_file,
            lr=model_cfg["lr"],
            epoch=model_cfg["epoch"],
            wordNgrams=model_cfg["wordNgrams"],
            dim=model_cfg["dim"],
            verbose=0,      # suppress fastText's own progress output
        )
    finally:
        os.unlink(train_file)   # always clean up the temp file

    return None, FastTextClassifier(model, label_classes)