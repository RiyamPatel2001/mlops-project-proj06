import numpy as np
from sklearn.cluster import DBSCAN


def cluster_user(user_data: dict, eps: float, min_samples: int) -> list[dict]:
    """
    Run DBSCAN on a single user's embeddings.

    Returns a list of cluster dicts (cluster_id is set by the caller):
        {
            "payees":          list[str],   deduplicated payees in this cluster
            "embeddings":      np.ndarray,  (k, 768) embeddings for these payees
            "existing_labels": list[str],   existing labels for context
        }

    Noise points (DBSCAN label == -1) are discarded.
    Returns [] when the user has fewer than min_samples transactions.
    """
    embeddings: np.ndarray = user_data["embeddings"]
    labels: list[str] = user_data["labels"]
    payees: list[str] = user_data["payees"]

    if len(payees) < min_samples:
        return []

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    cluster_labels = db.fit_predict(embeddings)

    clusters = []
    for cid in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == cid
        indices = [i for i, m in enumerate(mask) if m]

        raw_payees = [payees[i] for i in indices]
        cluster_embeddings = embeddings[mask]
        cluster_existing_labels = [labels[i] for i in indices]

        seen: set[str] = set()
        deduped_payees: list[str] = []
        for p in raw_payees:
            if p not in seen:
                seen.add(p)
                deduped_payees.append(p)

        clusters.append({
            "payees": deduped_payees,
            "embeddings": cluster_embeddings,
            "existing_labels": cluster_existing_labels,
        })

    return clusters
