"""
evaluate.py
───────────
Shared evaluation and MLflow logging for all model candidates.

Responsibility split:
  train.py      → mlflow.log_param()   (all config params, one place)
  evaluate.py   → mlflow.log_metric()  (weighted_f1, macro_f1, per-class F1)
                  mlflow.log_artifact() (classification_report.json + .txt)

evaluate_and_log() must be called inside an active mlflow.start_run() context.
"""

from __future__ import annotations

import json
import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def evaluate_and_log(
    clf,
    vec,
    X_val: pd.Series | list[str],
    y_val: np.ndarray,
    label_classes: list[str],
    config: dict,
) -> dict[str, float]:
    """
    Evaluate a fitted (vectorizer, classifier) pair on the validation set,
    log everything to the active MLflow run, and return the metric dict.

    Args:
        clf:           Fitted classifier (sklearn or compatible .predict())
        vec:           Fitted vectorizer with .transform(); pass None if
                       X_val is already a matrix (e.g. transformer embeddings)
        X_val:         Raw payee_norm strings, or pre-vectorized matrix
        y_val:         Integer-encoded ground-truth labels
        label_classes: Ordered list of category strings (from label_classes.json)
                       — index i corresponds to integer label i
        config:        Full config dict (logged as MLflow params)

    Returns:
        {"weighted_f1": float, "macro_f1": float}
    """

    # ── 1. Vectorize if needed ────────────────────────────────────────────────
    if vec is not None:
        X_vec = vec.transform(X_val)
    else:
        X_vec = X_val   # already a matrix

    # ── 2. Predict ────────────────────────────────────────────────────────────
    preds = clf.predict(X_vec)

    # ── 3. Aggregate metrics ──────────────────────────────────────────────────
    weighted_f1 = f1_score(y_val, preds, average="weighted")
    macro_f1    = f1_score(y_val, preds, average="macro")

    # ── 4. Per-class report ───────────────────────────────────────────────────
    all_labels = list(range(len(label_classes)))
    report_dict = classification_report(
        y_val,
        preds,
        labels=all_labels,
        target_names=label_classes,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_val,
        preds,
        labels=all_labels,
        target_names=label_classes,
        zero_division=0,
    )

    # ── 5. MLflow — aggregate metrics ────────────────────────────────────────
    model_name = config["model"]   # read-only, used for console output only
    mlflow.log_metric("weighted_f1", weighted_f1)
    mlflow.log_metric("macro_f1",    macro_f1)

    # Also log per-class F1 as individual metrics so the MLflow UI can sort
    for cat in label_classes:
        safe_name = cat.replace(" ", "_").replace("/", "_").replace("&", "and")
        mlflow.log_metric(f"f1_{safe_name}", report_dict[cat]["f1-score"])

    # ── 7. MLflow — artifacts ─────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:

        # 7a. Per-class report as JSON
        report_path = os.path.join(tmp, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact(report_path)

        # 7b. Per-class report as plain text (easier to read in UI)
        report_txt_path = os.path.join(tmp, "classification_report.txt")
        with open(report_txt_path, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_txt_path)

    # ── 8. Console summary ────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Model          : {model_name}")
    print(f"  Weighted F1    : {weighted_f1:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"{'─'*55}")
    print(report_str)

    # ── 9. Quality gate ───────────────────────────────────────────────────────
    gate_cfg = config.get("quality_gate", {})
    min_weighted = gate_cfg.get("weighted_f1", 0.75)
    min_macro    = gate_cfg.get("macro_f1",    0.55)

    passed = weighted_f1 >= min_weighted and macro_f1 >= min_macro
    status = "passed" if passed else "failed"
    mlflow.set_tag("quality_gate", status)
    print(
        f"[quality_gate] {status}  "
        f"(weighted_f1={weighted_f1:.4f}>={min_weighted}, "
        f"macro_f1={macro_f1:.4f}>={min_macro})"
    )
    if not passed:
        raise SystemExit(1)

    return {"weighted_f1": weighted_f1, "macro_f1": macro_f1}