"""
temporal_experiment.py
──────────────────────
Temporal generalisation experiment: train on 2022 data, evaluate on
2024 synthetic and real-world data to measure distribution shift.

MLflow experiment : layer1-temporal-generalization
Run name          : fasttext-2022-to-2024
"""

import json
import os
import sys
import tempfile
import time

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import utils as preprocess
from train import (
    load_config,
    setup_mlflow,
    get_git_sha,
    log_config_params,
    run_preprocessing,
    run_training,
    save_and_log_model,
)

# Load .env from project root (two levels up from training/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")


# ── MinIO helpers ─────────────────────────────────────────────────────────────

def _minio_client(cfg: dict):
    from minio import Minio

    endpoint = MINIO_ENDPOINT or cfg["minio"]["endpoint"]
    access   = MINIO_ACCESS_KEY or cfg["minio"].get("access_key", "minioadmin")
    secret   = MINIO_SECRET_KEY or cfg["minio"].get("secret_key", "minioadmin")

    endpoint_clean = endpoint.replace("http://", "").replace("https://", "")
    secure = endpoint.startswith("https://")
    return Minio(endpoint_clean, access_key=access, secret_key=secret, secure=secure)


def load_csv_from_minio(cfg: dict, object_path: str) -> pd.DataFrame:
    """Download object_path from the configured MinIO bucket and return as DataFrame."""
    client = _minio_client(cfg)
    bucket = cfg["minio"]["bucket"]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        client.fget_object(bucket, object_path, tmp_path)
        df = pd.read_csv(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return df


# ── Eval helpers ──────────────────────────────────────────────────────────────

def _encode_eval(df: pd.DataFrame, le: LabelEncoder) -> tuple[pd.Series, np.ndarray, int]:
    """
    Normalize payees and encode labels for an eval set using an already-fitted
    LabelEncoder.  Rows whose category is not in the training classes are dropped.

    Returns:
        X       — payee_norm Series
        y       — integer-encoded label array
        dropped — number of rows dropped due to unknown categories
    """
    df = df[["payee", "category"]].copy()
    df["payee_norm"] = df["payee"].apply(preprocess.normalize_payee)

    known_mask = df["category"].isin(le.classes_)
    dropped    = int((~known_mask).sum())
    df         = df[known_mask].reset_index(drop=True)

    df["label"] = le.transform(df["category"])
    return df["payee_norm"], df["label"].values, dropped


def _eval_and_log(
    clf,
    vec,
    X: pd.Series,
    y: np.ndarray,
    label_classes: list[str],
    suffix: str,
) -> dict[str, float]:
    """
    Run prediction, compute metrics, log to the active MLflow run with suffix.
    """
    X_vec = vec.transform(X) if vec is not None else X
    preds = clf.predict(X_vec)

    weighted_f1 = f1_score(y, preds, average="weighted", zero_division=0)
    macro_f1    = f1_score(y, preds, average="macro",    zero_division=0)

    mlflow.log_metric(f"weighted_f1{suffix}", weighted_f1)
    mlflow.log_metric(f"macro_f1{suffix}",    macro_f1)

    all_labels  = list(range(len(label_classes)))
    report_dict = classification_report(
        y, preds,
        labels=all_labels,
        target_names=label_classes,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y, preds,
        labels=all_labels,
        target_names=label_classes,
        zero_division=0,
    )

    for cat in label_classes:
        safe = cat.replace(" ", "_").replace("/", "_").replace("&", "and")
        mlflow.log_metric(f"f1_{safe}{suffix}", report_dict[cat]["f1-score"])

    with tempfile.TemporaryDirectory() as tmp:
        report_path = os.path.join(tmp, f"classification_report{suffix}.json")
        with open(report_path, "w") as fh:
            json.dump(report_dict, fh, indent=2)
        mlflow.log_artifact(report_path)

        report_txt = os.path.join(tmp, f"classification_report{suffix}.txt")
        with open(report_txt, "w") as fh:
            fh.write(report_str)
        mlflow.log_artifact(report_txt)

    print(f"\n{'─'*55}")
    print(f"  Eval set       : {suffix.lstrip('_')}")
    print(f"  Weighted F1    : {weighted_f1:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"{'─'*55}")
    print(report_str)

    return {"weighted_f1": weighted_f1, "macro_f1": macro_f1}


# ── Payee overlap ─────────────────────────────────────────────────────────────

def compute_payee_overlap(train_payees: pd.Series, eval_payees: pd.Series) -> dict:
    train_set = set(train_payees.apply(preprocess.normalize_payee))
    eval_set  = set(eval_payees.apply(preprocess.normalize_payee))
    shared    = train_set & eval_set
    new       = eval_set - train_set
    return {"shared_payees": len(shared), "new_payees": len(new)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    print(f"[config] Loaded {config_path}")

    # Force fastText for this experiment
    cfg["model"] = "fasttext"

    # Override experiment name
    cfg["mlflow"]["experiment_name"] = "layer1-temporal-generalization"

    setup_mlflow(cfg)

    with mlflow.start_run(run_name="fasttext-2022-to-2024"):
        mlflow.set_tag("git_sha",  get_git_sha())
        mlflow.set_tag("run_type", "temporal_generalization")

        log_config_params(cfg)

        # ── 1. Preprocess training data (2022) ────────────────────────────────
        print("[temporal] Preprocessing training data (2022) ...")
        X_train, X_val, y_train, y_val, label_classes = run_preprocessing(cfg)

        # Reconstruct the LabelEncoder fitted on training categories so we can
        # reuse it for eval sets without re-fitting.
        le = LabelEncoder()
        le.classes_ = np.array(label_classes)

        # ── 2. Train fastText ─────────────────────────────────────────────────
        print("[temporal] Training fastText ...")
        t0     = time.perf_counter()
        result = run_training(X_train, y_train, cfg)
        mlflow.log_metric("training_time_seconds", round(time.perf_counter() - t0, 2))

        vec, clf = (result if isinstance(result, tuple) and len(result) == 2
                    else (None, result))

        # ── 3. Load eval sets from MinIO ──────────────────────────────────────
        print("[temporal] Loading synthetic eval set (2024) from MinIO ...")
        df_synthetic = load_csv_from_minio(cfg, "data/raw/evaluation.csv")

        print("[temporal] Loading real-world eval set from MinIO ...")
        df_realworld = load_csv_from_minio(cfg, "data/raw/evaluation_moneydata.csv")

        # ── 4. Load training CSV for payee overlap computation ────────────────
        print("[temporal] Loading train.csv for payee overlap ...")
        df_train_raw = load_csv_from_minio(cfg, "data/raw/train.csv")

        # ── 5. Payee overlap (train vs 2024 synthetic) ────────────────────────
        overlap = compute_payee_overlap(df_train_raw["payee"], df_synthetic["payee"])
        mlflow.log_metric("shared_payees", overlap["shared_payees"])
        mlflow.log_metric("new_payees",    overlap["new_payees"])
        print(f"[temporal] Payee overlap — shared={overlap['shared_payees']}  "
              f"new={overlap['new_payees']}")

        with tempfile.TemporaryDirectory() as tmp:
            overlap_path = os.path.join(tmp, "payee_overlap.json")
            with open(overlap_path, "w") as fh:
                json.dump(overlap, fh, indent=2)
            mlflow.log_artifact(overlap_path)

        # ── 6. Encode eval sets ───────────────────────────────────────────────
        X_syn, y_syn, _ = _encode_eval(df_synthetic, le)

        X_rw, y_rw, dropped_rw = _encode_eval(df_realworld, le)
        mlflow.log_metric("realworld_dropped_rows", dropped_rw)
        if dropped_rw:
            print(f"[temporal] Dropped {dropped_rw} real-world rows "
                  f"with unknown categories.")

        # ── 7. Evaluate on both sets ──────────────────────────────────────────
        print("[temporal] Evaluating on synthetic (2024) set ...")
        metrics_syn = _eval_and_log(clf, vec, X_syn, y_syn, label_classes, "_synthetic")

        print("[temporal] Evaluating on real-world set ...")
        metrics_rw  = _eval_and_log(clf, vec, X_rw,  y_rw,  label_classes, "_realworld")

        # ── 8. Save model artifact ────────────────────────────────────────────
        save_and_log_model(vec, clf, cfg)

        print(
            f"\n[done] Temporal experiment complete — "
            f"synthetic_weighted_f1={metrics_syn['weighted_f1']:.4f}  "
            f"realworld_weighted_f1={metrics_rw['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
