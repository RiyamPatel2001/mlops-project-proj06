"""
temporal_experiment.py
──────────────────────
Time-based generalization experiment for the Layer-1 categorizer.

Splits the dataset into rolling monthly windows and tests whether a model
trained on earlier transactions generalizes to later ones — the key signal
for detecting data drift before it hits production.

For each fold:
  - Train  on all transactions  before  cutoff_date
  - Test   on transactions in  [cutoff_date, cutoff_date + window_months)
  - Log metrics + fold metadata as a nested MLflow child run

Usage:
    python3 temporal_experiment.py --config config.yaml
    python3 temporal_experiment.py --config config.yaml --window-months 3 --min-train-months 6
"""

import argparse
import os
import sys
import time
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
import train as train_module
import utils as preprocess
import evaluate as eval_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--window-months", type=int, default=1,
        help="Test window width in months (default: 1)",
    )
    parser.add_argument(
        "--min-train-months", type=int, default=6,
        help="Minimum months of history before the first test window (default: 6)",
    )
    return parser.parse_args()


def make_folds(
    df: pd.DataFrame,
    window_months: int,
    min_train_months: int,
):
    """Yield (df_train, df_test, cutoff_label) for each rolling window."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    min_date = df["date"].min()
    max_date = df["date"].max()
    cutoff   = min_date + relativedelta(months=min_train_months)

    while cutoff + relativedelta(months=window_months) <= max_date + timedelta(days=1):
        test_end = cutoff + relativedelta(months=window_months)
        df_train = df[df["date"] < cutoff].copy()
        df_test  = df[(df["date"] >= cutoff) & (df["date"] < test_end)].copy()

        if len(df_train) > 0 and len(df_test) > 0:
            yield df_train, df_test, cutoff.strftime("%Y-%m")

        cutoff = test_end


def run_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cfg: dict,
    cutoff_str: str,
) -> dict | None:
    """Train + evaluate one fold as a nested MLflow run. Returns metrics or None."""
    le = LabelEncoder()
    le.fit(df_train["category"])
    label_classes = le.classes_.tolist()

    # Restrict test to labels seen during training
    df_test = df_test[df_test["category"].isin(le.classes_)].copy()
    if len(df_test) == 0:
        print(f"[temporal] Fold {cutoff_str}: no test rows with known labels — skipping")
        return None

    for split in (df_train, df_test):
        if "payee_norm" not in split.columns:
            split["payee_norm"] = split["payee"].apply(preprocess.normalize_payee)

    df_train["label"] = le.transform(df_train["category"])
    df_test["label"]  = le.transform(df_test["category"])

    X_train = df_train["payee_norm"]
    y_train = df_train["label"].values
    X_test  = df_test["payee_norm"]
    y_test  = df_test["label"].values

    with mlflow.start_run(run_name=f"fold-{cutoff_str}", nested=True):
        mlflow.set_tag("fold_cutoff", cutoff_str)
        mlflow.log_param("train_rows", len(df_train))
        mlflow.log_param("test_rows",  len(df_test))
        mlflow.log_param("model",      cfg["model"])

        t0     = time.perf_counter()
        result = train_module.run_training(X_train, y_train, cfg)
        mlflow.log_metric("training_time_seconds", round(time.perf_counter() - t0, 2))

        vec, clf = result if isinstance(result, tuple) and len(result) == 2 else (None, result)

        # Catch SystemExit from the quality gate so a single bad fold doesn't
        # abort the whole experiment — it's already tagged in MLflow.
        try:
            metrics = eval_module.evaluate_and_log(
                clf=clf, vec=vec, X_val=X_test, y_val=y_test,
                label_classes=label_classes, config=cfg,
            )
        except SystemExit:
            metrics = None

    return metrics


def main() -> None:
    args = parse_args()
    cfg  = train_module.load_config(args.config)
    train_module.setup_mlflow(cfg)

    print(f"[temporal] Loading data from {cfg['data']['raw_path']} ...")
    df = pd.read_csv(cfg["data"]["raw_path"])
    print(f"[temporal] Loaded {len(df):,} rows")

    folds = list(make_folds(df, args.window_months, args.min_train_months))
    print(
        f"[temporal] {len(folds)} fold(s) — "
        f"window={args.window_months}mo  min_train={args.min_train_months}mo"
    )

    with mlflow.start_run(run_name=f"{cfg['model']}-temporal"):
        mlflow.set_tag("experiment_type", "temporal")
        mlflow.log_param("window_months",    args.window_months)
        mlflow.log_param("min_train_months", args.min_train_months)
        mlflow.log_param("n_folds",          len(folds))

        fold_metrics = []
        for df_train, df_test, cutoff_str in folds:
            print(f"\n[temporal] Fold {cutoff_str}: train={len(df_train):,}  test={len(df_test):,}")
            m = run_fold(df_train, df_test, cfg, cutoff_str)
            if m:
                fold_metrics.append(m)

        if fold_metrics:
            avg_weighted = float(np.mean([m["weighted_f1"] for m in fold_metrics]))
            avg_macro    = float(np.mean([m["macro_f1"]    for m in fold_metrics]))
            mlflow.log_metric("avg_weighted_f1", round(avg_weighted, 4))
            mlflow.log_metric("avg_macro_f1",    round(avg_macro,    4))

            print(f"\n{'─'*55}")
            print(f"  Temporal Experiment Summary ({len(fold_metrics)} folds)")
            print(f"  Avg Weighted F1 : {avg_weighted:.4f}")
            print(f"  Avg Macro F1    : {avg_macro:.4f}")
            print(f"{'─'*55}")


if __name__ == "__main__":
    main()
