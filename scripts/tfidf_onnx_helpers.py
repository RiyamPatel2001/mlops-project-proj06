"""
Train a word-level TF-IDF + LogisticRegression surrogate and export model.onnx.

Used by eval_tfidf_logreg.ipynb and by export_tfidf_onnx_surrogate.py (CLI).
Uses integer-encoded labels so skl2onnx conversion is reliable (string y often breaks).
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


def fit_surrogate_pipeline(csv_path: str) -> Pipeline:
    df = pd.read_csv(csv_path)
    if "payee" not in df.columns or "category" not in df.columns:
        raise ValueError("transactions.csv must contain payee and category columns")
    X = df["payee"].astype(str).fillna("")
    y_raw = df["category"].astype(str)
    y = LabelEncoder().fit_transform(y_raw)
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    max_features=8000,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    n_jobs=1,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _classifier_from_pipeline(pipe: Pipeline) -> LogisticRegression:
    for key in ("classifier", "clf", "model"):
        step = pipe.named_steps.get(key)
        if isinstance(step, LogisticRegression):
            return step
    for _, step in pipe.named_steps.items():
        if isinstance(step, LogisticRegression):
            return step
    raise ValueError("Pipeline has no LogisticRegression step (expected classifier/clf/model).")


def export_pipeline_to_onnx_file(pipe: Pipeline, path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    initial_type = [("input", StringTensorType([None, 1]))]
    last_err: Exception | None = None
    clf = _classifier_from_pipeline(pipe)
    # zipmap=False: probabilities as tensor (Triton ONNX backend rejects SEQUENCE / ZipMap).
    skl_opts = {id(clf): {"zipmap": False}, LogisticRegression: {"zipmap": False}}

    for opset in (17, 15, 13, 11):
        try:
            onnx_model = convert_sklearn(
                pipe,
                initial_types=initial_type,
                target_opset=opset,
                options=skl_opts,
            )
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"skl2onnx failed for all opsets (last error: {last_err!r})") from last_err


def resolve_transactions_csv(repo_root: str) -> str:
    """Find transactions.csv (Docker often lacks data/ if the file was never copied to the host)."""
    repo_root = os.path.abspath(repo_root)
    candidates = [
        os.path.join(repo_root, "data", "transactions.csv"),
        os.path.join(repo_root, "models", "transactions.csv"),
        os.path.join(repo_root, "transactions.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "transactions.csv not found. Put it in one of:\n  "
        + "\n  ".join(candidates)
        + "\nOn the host: copy from a machine that has the repo dataset, or download from your course materials. "
        "The filename is often gitignored so it is not created by git clone."
    )


def write_surrogate_onnx(repo_root: str) -> str:
    """Fit surrogate from transactions.csv; write models/tfidf_logreg_onnx/model.onnx."""
    repo_root = os.path.abspath(repo_root)
    csv_path = resolve_transactions_csv(repo_root)
    out_path = os.path.join(repo_root, "models", "tfidf_logreg_onnx", "model.onnx")
    pipe = fit_surrogate_pipeline(csv_path)
    export_pipeline_to_onnx_file(pipe, out_path)
    return out_path
