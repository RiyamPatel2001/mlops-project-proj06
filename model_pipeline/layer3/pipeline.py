import logging
import os
import pickle

import mlflow
import psycopg2
import yaml

from .cluster import cluster_user
from .namer import name_cluster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(config: dict) -> None:
    """
    Weekly orchestration: cluster every user's embeddings, name each cluster via
    the LLM, and write pending suggestions to Postgres.
    """
    store_path: str = config["layer3"]["store_path"]
    eps: float = config["layer3"]["eps"]
    min_samples: int = config["layer3"]["min_samples"]
    tracking_uri: str = config["mlflow"]["tracking_uri"].strip()

    if not os.path.exists(store_path):
        logger.warning("user_store.pkl not found at %s — nothing to process", store_path)
        return

    with open(store_path, "rb") as f:
        store: dict = pickle.load(f)

    dsn = os.environ.get("POSTGRES_DSN", config.get("postgres", {}).get("dsn", ""))
    if not dsn:
        raise RuntimeError(
            "Postgres DSN not set. Provide POSTGRES_DSN env var or set postgres.dsn in config.yaml."
        )
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

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
