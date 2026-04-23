#!/usr/bin/env python3
"""
training/eval_layer1.py
───────────────────────
Evaluate a trained Layer 1 model on an arbitrary eval CSV from MinIO.
Supports both fasttext and transformer (minilm / distilbert / mpnet) models.

Reuses evaluate_and_log() from evaluate.py — same metrics, MLflow artifacts,
and quality gate as a training run.  Pass --no-quality-gate for OOD datasets
(e.g. eval_moneydata.csv) where a lower score is expected.

Label classes
─────────────
  fasttext   : derived directly from the model (no file needed)
  transformer: resolved in order —
                 1. label_classes.json logged as MLflow artifact on the source run
                    (present when the model was trained after the save_and_log_model
                     change that logs it alongside the transformer weights)
                 2. data/processed/label_classes.json via Docker volume mount

Usage (inside GPU container or locally with GPU):
  python eval_layer1.py \\
    --run-id  <mlflow-run-id>   \\
    --model-type  minilm         \\
    --eval-csv    processed/eval_cex.csv \\
    --run-name    eval-minilm-cex

Docker — build (from training/):
  docker build -t categorizer-training-gpu -f Dockerfile.gpu .

Docker — run eval_cex.csv (from training/):
  docker run --rm --gpus all \\
    -v $(pwd)/../data:/app/data \\
    -e MLFLOW_TRACKING_URI=http://<FLOATING-IP>:30500 \\
    --entrypoint python \\
    categorizer-training-gpu \\
    eval_layer1.py \\
      --run-id    <run-id> \\
      --model-type minilm \\
      --eval-csv  processed/eval_cex.csv \\
      --run-name  eval-minilm-cex

Docker — run eval_moneydata.csv (OOD, disable quality gate):
  docker run --rm --gpus all \\
    -v $(pwd)/../data:/app/data \\
    -e MLFLOW_TRACKING_URI=http://<FLOATING-IP>:30500 \\
    --entrypoint python \\
    categorizer-training-gpu \\
    eval_layer1.py \\
      --run-id    <run-id> \\
      --model-type minilm \\
      --eval-csv  processed/eval_moneydata.csv \\
      --run-name  eval-minilm-moneydata \\
      --no-quality-gate
"""

import argparse
import copy
import io
import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
import utils as preprocess
import evaluate as eval_module


_TRANSFORMER_TYPES = {"minilm", "distilbert", "mpnet"}


# ── Fasttext wrapper ───────────────────────────────────────────────────────────

class _FasttextWrapper:
    """
    Makes fasttext.predict() return an integer array so evaluate_and_log()
    works without modification — same interface as TransformerClassifier.predict().
    """

    def __init__(self, model, label_classes: list[str]) -> None:
        self._model        = model
        self._label_to_int = {lbl: i for i, lbl in enumerate(label_classes)}

    def predict(self, X: "pd.Series | list[str]") -> np.ndarray:
        if isinstance(X, pd.Series):
            X = X.tolist()
        preds = []
        for text in X:
            labels, _ = self._model.predict(str(text), k=1)
            label = labels[0].replace("__label__", "").replace("_", " ")
            preds.append(self._label_to_int.get(label, 0))
        return np.array(preds)


# ── Loaders ────────────────────────────────────────────────────────────────────

def _label_classes_from_mlflow(run_id: str) -> list[str] | None:
    """Try to download label_classes.json from the source MLflow run."""
    try:
        path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/label_classes.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _label_classes_from_local(cfg: dict) -> list[str] | None:
    """Fall back to data/processed/label_classes.json (volume-mounted in Docker)."""
    path = os.path.join(cfg["data"]["processed_dir"], "label_classes.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_model_and_classes(
    run_id: str,
    model_type: str,
    cfg: dict,
) -> tuple[object, list[str]]:
    """
    Download model artifact from MLflow and return (clf, label_classes).

    fasttext   — label_classes derived from the model itself; no JSON needed.
    transformer — label_classes resolved from MLflow artifact then local fallback.
    """
    if model_type == "fasttext":
        import fasttext
        local_path    = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/fasttext.bin")
        raw           = fasttext.load_model(local_path)
        label_classes = sorted(
            lbl.replace("__label__", "").replace("_", " ")
            for lbl in raw.get_labels()
        )
        clf = _FasttextWrapper(raw, label_classes)
        return clf, label_classes

    # ── Transformer ────────────────────────────────────────────────────────────
    label_classes = _label_classes_from_mlflow(run_id)
    if label_classes is None:
        label_classes = _label_classes_from_local(cfg)
    if label_classes is None:
        raise FileNotFoundError(
            "label_classes.json not found in MLflow artifacts or at "
            f"{cfg['data']['processed_dir']}. Either mount data/processed/ "
            "with -v $(pwd)/../data:/app/data or retrain the model after the "
            "save_and_log_model fix that logs label_classes.json as an artifact."
        )

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from models.layer1.transformer_base import TransformerClassifier

    local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_type}")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Loading {model_type} on {device} from {local_path}")

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model     = AutoModelForSequenceClassification.from_pretrained(
        local_path,
        num_labels=len(label_classes),
        ignore_mismatched_sizes=True,
    ).to(device)

    max_length = cfg.get(model_type, {}).get("max_length", 64)
    clf = TransformerClassifier(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
    )
    return clf, label_classes


def _download_eval_csv(cfg: dict, object_path: str) -> pd.DataFrame:
    from minio import Minio
    raw_endpoint = os.environ.get("MINIO_ENDPOINT_URL", cfg["minio"]["endpoint"])
    endpoint     = raw_endpoint.replace("http://", "").replace("https://", "")
    secure       = raw_endpoint.startswith("https://")
    client = Minio(
        endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", cfg["minio"].get("access_key", "minioadmin")),
        secret_key=os.environ.get("MINIO_SECRET_KEY", cfg["minio"].get("secret_key", "minioadmin")),
        secure=secure,
    )
    response = client.get_object(cfg["minio"]["bucket"], object_path)
    df = pd.read_csv(io.BytesIO(response.read()))
    response.close()
    response.release_conn()
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone Layer 1 evaluation on an external CSV.")
    p.add_argument("--run-id",     required=True,
                   help="MLflow run ID to load the model artifact from")
    p.add_argument("--model-type", required=True,
                   choices=["fasttext", "minilm", "distilbert", "mpnet"])
    p.add_argument("--eval-csv",   required=True,
                   help="MinIO object path, e.g. processed/eval_cex.csv")
    p.add_argument("--run-name",   default=None,
                   help="MLflow run name (default: eval-<model-type>-<csv-stem>)")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--no-quality-gate", action="store_true",
                   help="Disable quality gate thresholds (recommended for OOD datasets)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    csv_stem = os.path.splitext(os.path.basename(args.eval_csv))[0]
    run_name = args.run_name or f"eval-{args.model_type}-{csv_stem}"

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].strip())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ── Load model + label classes ─────────────────────────────────────────────
    print(f"[eval] Loading {args.model_type} from run {args.run_id} ...")
    clf, label_classes = _load_model_and_classes(args.run_id, args.model_type, cfg)
    print(f"[eval] {len(label_classes)} label classes.")

    # ── Download eval CSV ──────────────────────────────────────────────────────
    print(f"[eval] Downloading {args.eval_csv} from MinIO ...")
    df = _download_eval_csv(cfg, args.eval_csv)
    print(f"[eval] {len(df):,} rows.")

    # ── Preprocess ────────────────────────────────────────────────────────────
    df["payee_norm"] = df["payee"].apply(preprocess.normalize_payee)

    known_mask = df["category"].isin(set(label_classes))
    dropped    = int((~known_mask).sum())
    df         = df[known_mask].reset_index(drop=True)
    if dropped:
        print(f"[eval] Dropped {dropped} rows with unknown categories. Remaining: {len(df):,}")

    le = LabelEncoder()
    le.classes_ = np.array(label_classes)
    y_val = le.transform(df["category"].tolist())
    X_val = df["payee_norm"]

    # ── Build eval config ─────────────────────────────────────────────────────
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["model"] = args.model_type
    if args.no_quality_gate:
        eval_cfg["quality_gate"] = {"weighted_f1": 0.0, "macro_f1": 0.0}

    # ── Evaluate + log ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("eval_csv",      args.eval_csv)
        mlflow.set_tag("model_type",    args.model_type)
        mlflow.set_tag("source_run_id", args.run_id)
        mlflow.log_param("source_run_id", args.run_id)
        mlflow.log_param("eval_csv",      args.eval_csv)
        mlflow.log_param("dropped_rows",  dropped)

        eval_module.evaluate_and_log(
            clf=clf,
            vec=None,
            X_val=X_val,
            y_val=y_val,
            label_classes=label_classes,
            config=eval_cfg,
        )

    print(f"\n[done] Evaluation logged under run '{run_name}'")


if __name__ == "__main__":
    main()
