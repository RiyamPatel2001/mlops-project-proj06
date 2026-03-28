"""
train.py
────────
Single entrypoint for all Layer-1 model training runs.
Controlled entirely by config.yaml — no hardcoded values here.

Usage:
    python3 train.py --config config.yaml

What it does (in order):
    1.  Parse --config argument
    2.  Load config.yaml
    3.  Set MLflow tracking URI + experiment name
    4.  Start MLflow run (named after the model)
    5.  Log all config params to MLflow
    6.  Preprocess: normalize payees, split by user, encode labels
    7.  Train: dispatch to the model module named in config["model"]
    8.  Evaluate: log weighted_f1, macro_f1, per-class F1, report artifact
    9.  Save model artifact (joblib)
    10. Log model artifact to MLflow
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

# ── Local modules (same package) ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import preprocess
import evaluate as eval_module

# Model registry — add new candidates here as you build them
MODEL_REGISTRY = {
    "tfidf_logreg": "models.tfidf_logreg",
    "fasttext":     "models.fasttext_model",
    "minilm":       "models.minilm_finetune",
    "distilbert":   "models.distilbert_finetune",
}


# ── 1. Argument parsing ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Layer-1 transaction categorizer.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


# ── 2. Config loading ─────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── 3–5. MLflow setup + param logging ────────────────────────────────────────

def setup_mlflow(cfg: dict) -> None:
    # MLflow reads MLFLOW_TRACKING_URI from the environment natively.
    # We only call set_tracking_uri() as a fallback for local runs where
    # the env var isn't set and config.yaml holds the URI instead.
    # Priority: MLFLOW_TRACKING_URI env var > config.yaml
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    # Resolve whichever URI is now active and verify the server is reachable.
    # This prevents silent fallback to local file-based tracking (mlruns/)
    # when the server is down — fail loudly instead.
    uri = mlflow.get_tracking_uri()
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(uri, timeout=5)
        except Exception as e:
            raise RuntimeError(
                f"MLflow server not reachable at {uri}\n"
                f"  Start it with: mlflow server --host 0.0.0.0 --port 5000\n"
                f"  Or set MLFLOW_TRACKING_URI to the correct address.\n"
                f"  Original error: {e}"
            )
 
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    # Log the URI actually in effect (env var may differ from config.yaml value)
    cfg["mlflow"]["tracking_uri"] = uri


def log_config_params(cfg: dict) -> None:
    """Flatten config to MLflow params. Nested dicts become 'section.key'."""
    for key, value in cfg.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                mlflow.log_param(f"{key}.{subkey}", subvalue)
        else:
            mlflow.log_param(key, value)


# ── 6. Preprocessing ──────────────────────────────────────────────────────────

def run_preprocessing(cfg: dict) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, list[str]]:
    """
    Load raw data, normalize payees, split by user, encode labels.

    Returns:
        X_train, X_val  — payee_norm string Series
        y_train, y_val  — integer-encoded label arrays
        label_classes   — ordered list of category strings (index == label int)
    """
    data_cfg = cfg["data"]

    df = pd.read_csv(data_cfg["raw_path"])
    print(f"[preprocess] Loaded {len(df):,} rows")

    # Normalize payees
    df["payee_norm"] = df["payee"].apply(preprocess.normalize_payee)

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])
    label_classes = le.classes_.tolist()

    # User-stratified split — both keys are required in config.yaml
    df_train, df_val = preprocess.user_stratified_split(
        df,
        val_frac=cfg["val_frac"],
        random_state=cfg["random_state"],
    )
    print(
        f"[preprocess] Train: {len(df_train):,} rows "
        f"({df_train['user_id'].nunique()} users) | "
        f"Val: {len(df_val):,} rows "
        f"({df_val['user_id'].nunique()} users)"
    )

    # Save processed splits so other pipeline stages can consume them
    processed_dir = data_cfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    df_train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(processed_dir, "val.csv"),   index=False)

    # Save label classes for serving layer
    with open(os.path.join(processed_dir, "label_classes.json"), "w") as f:
        json.dump(label_classes, f, indent=2)

    X_train = df_train["payee_norm"]
    X_val   = df_val["payee_norm"]
    y_train = df_train["label"].values
    y_val   = df_val["label"].values

    return X_train, X_val, y_train, y_val, label_classes


# ── 7. Model dispatch ─────────────────────────────────────────────────────────

def run_training(X_train, y_train, cfg: dict):
    """
    Import the correct model module from MODEL_REGISTRY and call train().
    Returns whatever train() returns — (vectorizer, classifier) for sklearn
    models, or the equivalent for future candidates.
    """
    model_name = cfg["model"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    import importlib
    module = importlib.import_module(MODEL_REGISTRY[model_name])
    print(f"[train] Fitting {model_name} ...")
    result = module.train(X_train, y_train, cfg)
    print(f"[train] Done.")
    return result


# ── 9–10. Artifact saving + logging ──────────────────────────────────────────

def save_and_log_model(vec, clf, cfg: dict) -> None:
    """
    Persist the fitted (vectorizer, classifier) pair with joblib,
    then log the file as an MLflow artifact.

    The artifact name encodes the model name so the MLflow UI is readable
    when multiple candidates are in the same experiment.
    """
    model_name = cfg["model"]
    output_dir = cfg["model_output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if model_name == "fasttext":
        # fasttext has its own binary format — save the underlying C++ model
        artifact_path = os.path.join(output_dir, f"{model_name}.bin")
        clf._model.save_model(artifact_path)
    elif model_name in ["minilm", "distilbert"]:
        # HuggingFace transformers — save in their native format
        artifact_path = os.path.join(output_dir, model_name)
        clf.save(artifact_path)
    else:
        # sklearn-compatible models — joblib pickle of {vectorizer, classifier}
        artifact_path = os.path.join(output_dir, f"{model_name}.joblib")
        joblib.dump({"vectorizer": vec, "classifier": clf}, artifact_path)
 
    print(f"[artifact] Saved model to {artifact_path}")
    mlflow.log_artifact(artifact_path)
    print(f"[artifact] Logged to MLflow.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Parse args
    args = parse_args()

    # 2. Load config
    cfg = load_config(args.config)
    print(f"[config] Loaded {args.config}  (model={cfg['model']})")

    # 3. MLflow setup
    setup_mlflow(cfg)

    # 4. Start run — named after the model for readable MLflow UI
    with mlflow.start_run(run_name=cfg["model"]):

        # 5. Log all config params up front
        log_config_params(cfg)

        # 6. Preprocess
        X_train, X_val, y_train, y_val, label_classes = run_preprocessing(cfg)

        # 7. Train — timed so we can log it as a metric
        t0 = time.perf_counter()
        result = run_training(X_train, y_train, cfg)
        training_time = time.perf_counter() - t0
        mlflow.log_metric("training_time_seconds", round(training_time, 2))
        print(f"[train] Training time: {training_time:.1f}s")

        # Unpack — sklearn models return (vectorizer, classifier)
        # Future models (fastText, transformers) may return (None, model)
        if isinstance(result, tuple) and len(result) == 2:
            vec, clf = result
        else:
            vec, clf = None, result

        # 8. Evaluate — logs metrics + artifacts to the active MLflow run
        metrics = eval_module.evaluate_and_log(
            clf=clf,
            vec=vec,
            X_val=X_val,
            y_val=y_val,
            label_classes=label_classes,
            config=cfg,
        )

        # 9–10. Save model + log artifact
        save_and_log_model(vec, clf, cfg)

        print(
            f"\n[done] Run complete — "
            f"weighted_f1={metrics['weighted_f1']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()