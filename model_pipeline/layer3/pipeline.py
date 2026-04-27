import logging
import os

import mlflow
import numpy as np
import psycopg2
import yaml

from .cluster import cluster_user
from .namer import name_cluster
from model_pipeline.layer2.user_store import load_store_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_store_from_postgres(cur) -> dict:
    """Build per-user store dict from layer2_examples (real production users)."""
    cur.execute("""
        SELECT user_id, payee, custom_category, embedding_vector
        FROM layer2_examples
        ORDER BY user_id, timestamp
    """)
    store = {}
    for user_id, payee, custom_category, embedding_vector in cur.fetchall():
        if user_id not in store:
            store[user_id] = {"embeddings": [], "labels": [], "payees": []}
        store[user_id]["embeddings"].append(embedding_vector)
        store[user_id]["labels"].append(custom_category)
        store[user_id]["payees"].append(payee)
    for uid in store:
        store[uid]["embeddings"] = np.array(store[uid]["embeddings"], dtype=np.float32)
    return store


def run_pipeline(config: dict) -> None:
    """
    Weekly orchestration: cluster every user's embeddings, name each cluster via
    the LLM, and write pending suggestions to Postgres.
    """
    eps: float = config["layer3"]["eps"]
    min_samples: int = config["layer3"]["min_samples"]
    tracking_uri: str = config["mlflow"]["tracking_uri"].strip()

    dsn = os.environ.get("POSTGRES_DSN", config.get("postgres", {}).get("dsn", ""))
    if not dsn:
        raise RuntimeError(
            "Postgres DSN not set. Provide POSTGRES_DSN env var or set postgres.dsn in config.yaml."
        )
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    source = os.environ.get("LAYER3_SOURCE", "postgres")
    if source == "minio":
        store = load_store_dict(config["layer3"]["store_path"])
        if not store:
            logger.warning("user_store.pkl not found at %s — nothing to process",
                           config["layer3"]["store_path"])
            cur.close()
            conn.close()
            return
    else:
        store = _load_store_from_postgres(cur)
        if not store:
            logger.warning("layer2_examples is empty — nothing to process")
            cur.close()
            conn.close()
            return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("layer3-clustering")

    total_users = 0
    total_clusters = 0
    total_suggestions = 0

    with mlflow.start_run():
        for user_id, user_data in store.items():
            if len(user_data.get("payees", [])) < min_samples:
                continue

            clusters = cluster_user(user_data, eps=eps, min_samples=min_samples)
            if not clusters:
                continue

            total_users += 1
            total_clusters += len(clusters)

            for i, cluster in enumerate(clusters):
                cluster_id = f"{user_id}_c{i}"
                suggested_name = name_cluster(cluster["payees"], cluster["existing_labels"])

                cur.execute(
                    """
                    INSERT INTO layer3_suggestions
                        (user_id, cluster_id, suggested_category_name, payee_list, status, created_at)
                    VALUES (%s, %s, %s, %s, 'pending', NOW())
                    ON CONFLICT (cluster_id) DO NOTHING
                    """,
                    (user_id, cluster_id, suggested_name, cluster["payees"]),
                )
                total_suggestions += 1
                logger.info("  user=%s cluster=%s name=%r payees=%d",
                            user_id, cluster_id, suggested_name, len(cluster["payees"]))

        conn.commit()
        cur.close()
        conn.close()

        mlflow.log_metrics({
            "total_users_processed": float(total_users),
            "total_clusters_found": float(total_clusters),
            "total_suggestions_written": float(total_suggestions),
        })

    logger.info("Pipeline complete — users=%d clusters=%d suggestions=%d",
                total_users, total_clusters, total_suggestions)


if __name__ == "__main__":
    _config_path = os.path.join(os.path.dirname(__file__), "..", "layer2", "config.yaml")
    with open(_config_path) as _f:
        _config = yaml.safe_load(_f)
    run_pipeline(_config)
