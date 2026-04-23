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
import subprocess
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
import utils as preprocess
import evaluate as eval_module

# Model registry — add new candidates here as you build them
MODEL_REGISTRY = {
    "tfidf_logreg": "models.layer1.tfidf_logreg",
    "fasttext":     "models.layer1.fasttext",
    "minilm":       "models.layer1.minilm_finetune",
    "distilbert":   "models.layer1.distilbert_finetune",
    "mpnet":        "models.layer1.mpnet_finetune",
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
    # Priority: DATA_RAW_PATH env var > config.yaml (mirrors MLFLOW_TRACKING_URI pattern)
    if raw_path := os.environ.get("DATA_RAW_PATH"):
        cfg["data"]["raw_path"] = raw_path
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


# Maps model-specific param names to canonical top-level names so that
# runs from different models align in the MLflow comparison table.
_PARAM_ALIASES = {
    "lr":    "learning_rate",   # fasttext -> canonical
    "epoch": "num_epochs",      # fasttext -> canonical
}


def get_git_sha() -> str:
    # Docker containers don't have .git (build context is training/ only).
    # Callers should pass GIT_SHA=$(git rev-parse HEAD) via -e at docker run time.
    if sha := os.environ.get("GIT_SHA", "").strip():
        return sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def log_config_params(cfg: dict) -> None:
    """
    Log config to MLflow with normalized param names.

    Top-level scalars (model, val_frac, random_state) are logged as-is.
    Only the active model's hyperparameters are logged — remapped to canonical
    names so they align across runs in the MLflow comparison table instead of
    producing sparse per-model columns.
    """
    for key, value in cfg.items():
        if not isinstance(value, dict):
            mlflow.log_param(key, value)

    model_params = cfg.get(cfg["model"], {})
    for key, value in model_params.items():
        mlflow.log_param(_PARAM_ALIASES.get(key, key), value)


# ── 6. Preprocessing ──────────────────────────────────────────────────────────

def _load_raw_csv(cfg: dict) -> pd.DataFrame:
    """Load raw CSV from a local path or an authenticated MinIO HTTP URL."""
    raw_path = cfg["data"]["raw_path"]
    if not raw_path.startswith(("http://", "https://")):
        return pd.read_csv(raw_path)

    from urllib.parse import urlparse
    from minio import Minio

    parsed = urlparse(raw_path)
    endpoint_clean = parsed.netloc
    secure = parsed.scheme == "https"
    bucket, object_key = parsed.path.lstrip("/").split("/", 1)

    minio_cfg = cfg.get("minio", {})
    client = Minio(
        endpoint_clean,
        access_key=os.environ.get("MINIO_ACCESS_KEY", minio_cfg.get("access_key", "minioadmin")),
        secret_key=os.environ.get("MINIO_SECRET_KEY", minio_cfg.get("secret_key", "minioadmin123")),
        secure=secure,
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        client.fget_object(bucket, object_key, tmp_path)
        return pd.read_csv(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_preprocessing(cfg: dict) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, list[str]]:
    """
    Load raw data, normalize payees, split by user, encode labels.

    Returns:
        X_train, X_val  — payee_norm string Series
        y_train, y_val  — integer-encoded label arrays
        label_classes   — ordered list of category strings (index == label int)
    """
    data_cfg = cfg["data"]

    df = _load_raw_csv(cfg)
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

    upload_processed_to_minio(cfg, processed_dir)

    return X_train, X_val, y_train, y_val, label_classes


# ── 6b. Upload processed splits to MinIO ─────────────────────────────────────

def upload_processed_to_minio(cfg: dict, processed_dir: str) -> None:
    try:
        from minio import Minio
    except ImportError:
        print("[minio] WARNING: minio package not installed — skipping upload.")
        return

    minio_cfg = cfg.get("minio", {})
    if not minio_cfg:
        print("[minio] No minio config found — skipping upload.")
        return

    try:
        endpoint_url = os.environ.get("MINIO_ENDPOINT_URL", minio_cfg["endpoint"])
        endpoint = endpoint_url.replace("http://", "").replace("https://", "")
        secure   = endpoint_url.startswith("https://")
        client   = Minio(
            endpoint,
            access_key=os.environ.get("MINIO_ACCESS_KEY", minio_cfg.get("access_key", "minioadmin")),
            secret_key=os.environ.get("MINIO_SECRET_KEY", minio_cfg.get("secret_key", "minioadmin")),
            secure=secure,
        )

        bucket = minio_cfg["bucket"]

        for filename in ("train.csv", "val.csv", "label_classes.json"):
            local_path = os.path.join(processed_dir, filename)
            if not os.path.exists(local_path):
                continue
            object_name = f"data/processed/{filename}"
            client.fput_object(bucket, object_name, local_path)
            print(f"[minio] Uploaded {filename} → {bucket}/{object_name}")

    except Exception as e:
        print(f"[minio] WARNING: upload failed — {e}. Continuing without MinIO upload.")


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

def _direct_minio_upload(local_path: str, cfg: dict) -> None:
    """
    Upload a file directly to MinIO at the path MLflow expects for the active run,
    bypassing the MLflow artifact proxy.

    With --serve-artifacts, Gunicorn buffers the entire POST body before forwarding
    to MinIO. For fasttext.bin (~764 MB) this exceeds the pod's available memory
    and causes repeated 500s. Writing directly via the minio SDK sidesteps the
    proxy while keeping the file at the path mlflow.artifacts.download_artifacts()
    resolves to on download.

    S3 key: <experiment_id>/<run_id>/artifacts/<basename>
    Bucket: mlflow-artifacts  (matches --artifacts-destination in deployment.yaml)
    """
    from minio import Minio

    run           = mlflow.active_run()
    experiment_id = run.info.experiment_id
    run_id        = run.info.run_id
    filename      = os.path.basename(local_path)
    object_name   = f"{experiment_id}/{run_id}/artifacts/{filename}"
    bucket        = "mlflow-artifacts"

    raw_endpoint = os.environ.get("MINIO_ENDPOINT_URL", cfg["minio"]["endpoint"])
    endpoint     = raw_endpoint.replace("http://", "").replace("https://", "")
    secure       = raw_endpoint.startswith("https://")

    client = Minio(
        endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", cfg["minio"].get("access_key", "minioadmin")),
        secret_key=os.environ.get("MINIO_SECRET_KEY", cfg["minio"].get("secret_key", "minioadmin")),
        secure=secure,
    )

    file_mb = os.path.getsize(local_path) / 1_048_576
    print(f"[artifact] Direct-uploading {filename} ({file_mb:.0f} MB) → {bucket}/{object_name}")
    client.fput_object(bucket, object_name, local_path)
    print(f"[artifact] Upload complete.")


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
        print(f"[artifact] Saved model to {artifact_path}")
        # ~764 MB binary — upload directly to MinIO to bypass the MLflow proxy
        _direct_minio_upload(artifact_path, cfg)
    elif model_name in ["minilm", "distilbert", "mpnet"]:
        # HuggingFace transformers — save in their native format
        artifact_path = os.path.join(output_dir, model_name)
        clf.save(artifact_path)
        print(f"[artifact] Saved model to {artifact_path}")
        mlflow.log_artifact(artifact_path)
        # Transformers carry no label names — log label_classes.json alongside
        # the model so eval_layer1.py can resolve the integer→category mapping.
        lc_path = os.path.join(cfg["data"]["processed_dir"], "label_classes.json")
        if os.path.exists(lc_path):
            mlflow.log_artifact(lc_path)
            print(f"[artifact] Logged label_classes.json to MLflow.")
        else:
            print(f"[artifact] WARNING: {lc_path} not found — eval_layer1.py will need a volume mount.")
        print(f"[artifact] Logged to MLflow.")
    else:
        # sklearn-compatible models — joblib pickle of {vectorizer, classifier}
        artifact_path = os.path.join(output_dir, f"{model_name}.joblib")
        joblib.dump({"vectorizer": vec, "classifier": clf}, artifact_path)
        print(f"[artifact] Saved model to {artifact_path}")
        mlflow.log_artifact(artifact_path)
        print(f"[artifact] Logged to MLflow.")


# ── 11. MLflow Model Registry ────────────────────────────────────────────────

# Mirrors retrain.py's _MODEL_ARTIFACT_PATHS — artifact path relative to run root
_MODEL_ARTIFACT_PATHS = {
    "tfidf_logreg": "tfidf_logreg.joblib",
    "fasttext":     "fasttext.bin",
    "minilm":       "minilm",
    "distilbert":   "distilbert",
    "mpnet":        "mpnet",
}


def register_model_if_passed(cfg: dict, run_id: str) -> None:
    """
    Register the logged model artifact to the MLflow Model Registry
    if the quality gate passed for this run.
    Only registers if quality_gate tag == "passed" on the active run.
    Skips silently if the MLflow server does not support the registry
    (e.g. file-based tracking URI).
    """
    active_run = mlflow.active_run()
    gate_status = active_run.data.tags.get("quality_gate", "")
    if gate_status != "passed":
        print(f"[registry] Skipping registration — quality_gate={gate_status!r}")
        return

    model_name = cfg["model"]
    registered_model_name = cfg.get("mlflow", {}).get(
        "model_name", "transaction-categorizer"
    )
    artifact_path = _MODEL_ARTIFACT_PATHS.get(model_name)
    if artifact_path is None:
        print(
            f"[registry] WARNING: No artifact path mapping for model '{model_name}'"
            " — skipping registration."
        )
        return

    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

        # Annotate the registered version with key metadata for traceability
        metrics = active_run.data.metrics
        tags = active_run.data.tags
        wf1 = metrics.get("weighted_f1")
        mf1 = metrics.get("macro_f1")
        dataset_version = tags.get("dataset_version", "")

        desc_parts = []
        if dataset_version:
            desc_parts.append(f"dataset_version={dataset_version}")
        if wf1 is not None:
            desc_parts.append(f"weighted_f1={wf1:.4f}")
        if mf1 is not None:
            desc_parts.append(f"macro_f1={mf1:.4f}")

        mlflow.tracking.MlflowClient().update_model_version(
            name=registered_model_name,
            version=mv.version,
            description=", ".join(desc_parts) if desc_parts else "no metadata available",
        )
        print(
            f"[registry] Registered {registered_model_name} version {mv.version}"
            f" (run_id={run_id})"
        )
    except Exception as exc:
        print(f"[registry] WARNING: Model registration failed — {exc}. Continuing.")


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

        # Tag the run with the git SHA so every run is traceable to a commit
        mlflow.set_tag("git_sha", get_git_sha())

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

        # 11. Register to MLflow Model Registry if quality gate passed
        run_id = mlflow.active_run().info.run_id
        register_model_if_passed(cfg, run_id)

        print(
            f"\n[done] Run complete — "
            f"weighted_f1={metrics['weighted_f1']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
