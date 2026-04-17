"""
retrain.py
──────────
Triggered retraining entrypoint for the Layer-1 categorizer.

Workflow:
  1. Download new labeled CSVs from MinIO (data.retraining_prefix)
  2. Merge with the base raw dataset (data.raw_path) — deduped by transaction_id
  3. Run the full train → evaluate → upload-processed pipeline
  4. Quality gate in evaluate.py raises SystemExit(1) on regression

Usage:
    python3 retrain.py --config config.yaml
    python3 retrain.py --config config.yaml --no-merge
"""

import argparse
import io
import os
import sys
import tempfile
import time

import mlflow
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import train as train_module
import evaluate as eval_module
from minio import Minio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain Layer-1 categorizer with new data from MinIO.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging with the base raw dataset; use retraining data only.",
    )
    return parser.parse_args()


def make_minio_client(cfg: dict) -> Minio:
    minio_cfg = cfg["minio"]
    endpoint = minio_cfg["endpoint"].replace("http://", "").replace("https://", "")
    secure   = minio_cfg["endpoint"].startswith("https://")
    return Minio(
        endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=secure,
    )


def download_retraining_data(client: Minio, cfg: dict) -> pd.DataFrame:
    bucket = cfg["minio"]["bucket"]
    prefix = cfg["data"]["retraining_prefix"].rstrip("/") + "/"

    objects  = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    csv_objs = [o for o in objects if o.object_name.endswith(".csv")]

    if not csv_objs:
        raise RuntimeError(f"No CSV files found in MinIO {bucket}/{prefix}")

    frames = []
    for obj in csv_objs:
        print(f"[retrain] Downloading {obj.object_name} ...")
        response = client.get_object(bucket, obj.object_name)
        frames.append(pd.read_csv(io.BytesIO(response.read())))
        response.close()
        response.release_conn()

    df = pd.concat(frames, ignore_index=True)
    print(f"[retrain] {len(df):,} retraining rows from {len(csv_objs)} file(s)")
    return df


def main() -> None:
    args = parse_args()
    cfg  = train_module.load_config(args.config)

    client   = make_minio_client(cfg)
    df_new   = download_retraining_data(client, cfg)

    if not args.no_merge:
        base_path = cfg["data"]["raw_path"]
        try:
            df_base     = pd.read_csv(base_path)
            df_combined = pd.concat([df_base, df_new], ignore_index=True)
            if "transaction_id" in df_combined.columns:
                df_combined = df_combined.drop_duplicates(subset=["transaction_id"])
            print(
                f"[retrain] Merged base ({len(df_base):,}) + new ({len(df_new):,})"
                f" → {len(df_combined):,} rows"
            )
        except Exception as exc:
            print(f"[retrain] Could not load base dataset ({exc}); using retraining data only.")
            df_combined = df_new
    else:
        df_combined = df_new

    with tempfile.TemporaryDirectory() as tmp:
        merged_path = os.path.join(tmp, "retrain_raw.csv")
        df_combined.to_csv(merged_path, index=False)
        cfg["data"]["raw_path"] = merged_path

        train_module.setup_mlflow(cfg)

        with mlflow.start_run(run_name=f"{cfg['model']}-retrain"):
            mlflow.set_tag("git_sha", train_module.get_git_sha())
            mlflow.set_tag("trigger", "retrain")
            train_module.log_config_params(cfg)

            X_train, X_val, y_train, y_val, label_classes = train_module.run_preprocessing(cfg)

            t0     = time.perf_counter()
            result = train_module.run_training(X_train, y_train, cfg)
            mlflow.log_metric("training_time_seconds", round(time.perf_counter() - t0, 2))

            vec, clf = result if isinstance(result, tuple) and len(result) == 2 else (None, result)

            metrics = eval_module.evaluate_and_log(
                clf=clf, vec=vec, X_val=X_val, y_val=y_val,
                label_classes=label_classes, config=cfg,
            )

            train_module.save_and_log_model(vec, clf, cfg)

            print(
                f"\n[done] Retrain complete — "
                f"weighted_f1={metrics['weighted_f1']:.4f}  "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )


if __name__ == "__main__":
    main()
