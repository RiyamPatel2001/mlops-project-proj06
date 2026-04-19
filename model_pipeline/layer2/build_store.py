"""
model_pipeline/layer2/build_store.py

Offline script. Runs once at "training time" to build and persist the user store.

Reads:
  - transactions_2022.csv (all rows)
  - transactions_2023.csv (first 70% of each user's transactions, chronologically)

Drops irrelevant columns: newid, diary_newid, survey_source.
Batch-embeds all payee strings with all-mpnet-base-v2.
Builds per-user store and saves as user_store.pkl.
Logs the artifact to MLflow.

Usage:
    python -m model_pipeline.layer2.build_store
"""

import io
import os
import pickle
import yaml
import numpy as np
import pandas as pd
import mlflow
from minio import Minio

from model_pipeline.layer2.embedder import Embedder

EXTRA_COLS = ["newid", "diary_newid", "survey_source"]


_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str = _DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_minio_client(cfg: dict) -> Minio:
    endpoint = cfg["minio"]["endpoint"].replace("http://", "").replace("https://", "")
    secure = cfg["minio"]["endpoint"].startswith("https://")
    return Minio(
        endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=secure,
    )


def load_csv(cfg: dict) -> pd.DataFrame:
    client = make_minio_client(cfg)
    bucket = cfg["minio"]["bucket"]
    obj = cfg["minio"]["object"]
    response = client.get_object(bucket, obj)
    df = pd.read_csv(io.BytesIO(response.read()))
    response.close()
    response.release_conn()
    df.drop(columns=[c for c in EXTRA_COLS if c in df.columns], inplace=True)
    return df


def first_n_percent_per_user(df: pd.DataFrame, pct: float = 0.70) -> pd.DataFrame:
    """Return the first `pct` of each user's rows, sorted chronologically."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["user_id", "date"], inplace=True)
    result = (
        df.groupby("user_id", group_keys=False)
        .apply(lambda g: g.iloc[: max(1, int(len(g) * pct))])
    )
    return result.reset_index(drop=True)


def build_store(df: pd.DataFrame, embedder: Embedder) -> dict:
    """
    Given a DataFrame with columns [user_id, payee, category],
    batch-embed all payees and build the per-user dictionary.
    """
    payees = df["payee"].tolist()
    print(f"  Embedding {len(payees)} transactions...")
    embeddings = embedder.embed_batch(payees)  # (n, 768) unit-normalized

    store = {}
    for idx, row in enumerate(df.itertuples(index=False)):
        uid = row.user_id
        if uid not in store:
            store[uid] = {
                "embeddings": [],
                "labels": [],
                "payees": [],
            }
        store[uid]["embeddings"].append(embeddings[idx])
        store[uid]["labels"].append(row.category)
        store[uid]["payees"].append(row.payee)

    # Convert lists to numpy arrays
    for uid in store:
        store[uid]["embeddings"] = np.array(store[uid]["embeddings"], dtype=np.float32)

    return store


def main():
    cfg = load_config()
    l2 = cfg["layer2"]

    output_path = l2["store_path"]
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    print(f"Loading data from {cfg['minio']['endpoint']}/{cfg['minio']['bucket']}/{cfg['minio']['object']} ...")
    df = load_csv(cfg)
    print(f"Loaded {len(df):,} rows")

    # Use first 70% of each user's history so the last 30% stays unseen for evaluation
    df = first_n_percent_per_user(df, pct=0.70)
    print(f"Building store from {len(df):,} rows (first 70% per user by date)")

    print(f"Loading embedder: {l2['model_name']}")
    embedder = Embedder(model_name=l2["model_name"], max_length=l2.get("max_length", 128))

    print("Building store...")
    store = build_store(df, embedder)
    print(f"Built store for {len(store)} users.")

    with open(output_path, "wb") as f:
        pickle.dump(store, f)
    print(f"Saved store to {output_path}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].strip())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="build_store"):
        mlflow.log_param("layer2.model_name", l2["model_name"])
        mlflow.log_param("layer2.num_users", len(store))
        mlflow.log_param("layer2.store_transactions", len(df))
        mlflow.log_artifact(output_path)
        print("Logged artifact to MLflow.")


if __name__ == "__main__":
    main()