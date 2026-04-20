"""
retrain.py
──────────
Triggered retraining entrypoint for the Layer-1 categorizer.
Runs weekly as a K8s CronJob.

Workflow:
  1. Load config + .env
  2. Discover the latest retraining dataset in MinIO (retraining/retraining_dataset_v*.csv)
  3. Download it to a temp file and point cfg["data"]["raw_path"] at it
  4. Retrain → evaluate → save, all inside a single MLflow run
  5. Clean up temp file

Quality gate is enforced by evaluate_and_log() — raises SystemExit(1) on failure.
No retraining files → exit 0 (expected during initial deployment).

Usage:
    python3 retrain.py --config config.yaml
    python3 retrain.py --config config.yaml --no-merge
"""

import argparse
import os
import sys
import time
import tempfile
from datetime import datetime

import mlflow
from dotenv import load_dotenv
from minio import Minio

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import train as train_module
import evaluate as eval_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain Layer-1 categorizer with latest MinIO dataset.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def make_minio_client(cfg: dict) -> Minio:
    minio_cfg = cfg["minio"]
    endpoint  = minio_cfg["endpoint"].replace("http://", "").replace("https://", "")
    secure    = minio_cfg["endpoint"].startswith("https://")
    return Minio(
        endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin123"),
        secure=secure,
    )


def discover_latest_dataset(client: Minio, cfg: dict) -> str | None:
    """
    List all retraining_dataset_v*.csv objects under retraining/ in MinIO.
    Returns the object name of the latest file (lexicographic sort on filename),
    or None if none exist yet.
    """
    bucket = cfg["minio"]["bucket"]
    prefix = "retraining/"

    objects  = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    datasets = [
        o.object_name for o in objects
        if os.path.basename(o.object_name).startswith("retraining_dataset_v")
        and o.object_name.endswith(".csv")
    ]

    if not datasets:
        return None

    datasets.sort()
    return datasets[-1]


def download_dataset(client: Minio, cfg: dict, object_name: str) -> str:
    """Download object_name from MinIO to a local temp file. Returns temp path."""
    bucket   = cfg["minio"]["bucket"]
    tmp_path = tempfile.mktemp(suffix=".csv")
    client.fget_object(bucket, object_name, tmp_path)
    print(f"[retrain] Downloaded {object_name} → {tmp_path}")
    return tmp_path


def main() -> None:
    args = parse_args()

    # ── Step 1: Load config + .env ────────────────────────────────────────────
    # .env lives at the project root (one level above training/)
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=dotenv_path)

    cfg = train_module.load_config(args.config)
    print(f"[retrain] Config loaded  (model={cfg['model']})")

    # Override experiment so retraining runs stay separate from sweep runs
    cfg["mlflow"]["experiment_name"] = "layer1-retraining"
    train_module.setup_mlflow(cfg)

    # ── Step 2: Discover latest retraining dataset ────────────────────────────
    client      = make_minio_client(cfg)
    object_name = discover_latest_dataset(client, cfg)

    if object_name is None:
        print(
            "[retrain] No retraining datasets found in MinIO under retraining/.\n"
            "          This is expected before any feedback has been collected.\n"
            "          Nothing to do — exiting cleanly."
        )
        sys.exit(0)

    dataset_filename = os.path.basename(object_name)
    print(f"[retrain] Using dataset: {dataset_filename}")

    # ── Step 3: Download to temp file ─────────────────────────────────────────
    tmp_path = download_dataset(client, cfg, object_name)
    cfg["data"]["raw_path"] = tmp_path

    # ── Steps 4 + 5: Retrain inside try/finally for guaranteed cleanup ────────
    try:
        model_name = cfg["model"]
        run_name   = f"retrain-{model_name}-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("run_type",           "retraining")
            mlflow.set_tag("retraining_dataset", dataset_filename)
            mlflow.set_tag("git_sha",            train_module.get_git_sha())
            train_module.log_config_params(cfg)

            # Preprocess — writes splits locally + uploads to MinIO data/processed/
            X_train, X_val, y_train, y_val, label_classes = train_module.run_preprocessing(cfg)

            # Train
            t0     = time.perf_counter()
            result = train_module.run_training(X_train, y_train, cfg)
            mlflow.log_metric("training_time_seconds", round(time.perf_counter() - t0, 2))

            vec, clf = result if isinstance(result, tuple) and len(result) == 2 else (None, result)

            # Evaluate — sets quality_gate tag, raises SystemExit(1) on failure
            metrics = eval_module.evaluate_and_log(
                clf=clf, vec=vec, X_val=X_val, y_val=y_val,
                label_classes=label_classes, config=cfg,
            )

            # Save + log artifact
            train_module.save_and_log_model(vec, clf, cfg)

            quality_gate = mlflow.active_run().data.tags.get("quality_gate", "unknown")

        # ── Step 6: Summary ───────────────────────────────────────────────────
        print(
            f"\n{'═'*55}\n"
            f"  Dataset        : {dataset_filename}\n"
            f"  Model          : {model_name}\n"
            f"  Weighted F1    : {metrics['weighted_f1']:.4f}\n"
            f"  Macro F1       : {metrics['macro_f1']:.4f}\n"
            f"  Quality gate   : {quality_gate}\n"
            f"{'═'*55}"
        )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"[retrain] Cleaned up temp file {tmp_path}")


if __name__ == "__main__":
    main()
