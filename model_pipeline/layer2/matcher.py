"""
layer2/matcher.py

Pure computation, no I/O. All embeddings must be unit-normalized before calling
these functions — cosine similarity then reduces to a dot product.
"""

import numpy as np
from collections import Counter
from typing import List, Tuple


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query vector and all rows in matrix.

    Args:
        query:  (768,) unit-normalized embedding
        matrix: (n, 768) unit-normalized embeddings

    Returns:
        (n,) similarity scores in [-1, 1]
    """
    return matrix @ query  # dot product == cosine sim for unit vectors


def get_top_k(
    query: np.ndarray,
    user_data: dict,
    k: int,
) -> List[Tuple[str, float]]:
    """
    Return top-k most similar stored transactions.

    Args:
        query:     (768,) unit-normalized query embedding
        user_data: dict with keys "embeddings" (n, 768), "labels" (list), "payees" (list)
        k:         number of neighbors to return

    Returns:
        List of (category, similarity_score) tuples sorted by similarity descending.
    """
    embeddings = user_data["embeddings"]  # (n, 768)
    labels = user_data["labels"]

    scores = cosine_similarity(query, embeddings)  # (n,)
    k_actual = min(k, len(scores))
    top_indices = np.argpartition(scores, -k_actual)[-k_actual:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [(labels[i], float(scores[i])) for i in top_indices]


def majority_vote(
    neighbors: List[Tuple[str, float]],
    threshold: float,
) -> Tuple[str, float, bool]:
    """
    Aggregate top-k neighbors into a single prediction.

    Args:
        neighbors: list of (category, similarity_score) from get_top_k
        threshold: minimum similarity score to trust Layer 2

    Returns:
        (majority_category, max_similarity_score, threshold_exceeded)
    """
    if not neighbors:
        return ("", 0.0, False)

    max_score = neighbors[0][1]  # already sorted descending
    categories = [cat for cat, _ in neighbors]
    majority_category = Counter(categories).most_common(1)[0][0]
    threshold_exceeded = max_score >= threshold

    return (majority_category, max_score, threshold_exceeded)