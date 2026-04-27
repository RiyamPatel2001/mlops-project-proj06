"""
Layer 2 — personal history KNN override.

For a given user + payee string:
1. Call Saketh's external embedding service to get the MiniLM vector.
2. Retrieve the user's stored examples from Postgres.
3. Compute cosine similarity between the incoming vector and each stored vector.
4. If the top match >= threshold, do a majority vote among top-k neighbours.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from typing import Optional

import httpx
import numpy as np

from app import db
from app.config import (
    EMBEDDING_SERVICE_URL,
    LAYER2_MIN_EXAMPLES,
    LAYER2_SIMILARITY_THRESHOLD,
    LAYER2_TOP_K,
)
from app.feature_computation import normalize_payee

logger = logging.getLogger(__name__)


async def get_embedding(text: str) -> Optional[list[float]]:
    """Call the external embedding micro-service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{EMBEDDING_SERVICE_URL}/embed",
                json={"text": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding") or data.get("vector")
            if embedding:
                return embedding
    except Exception as exc:
        logger.warning("Embedding service unavailable, using local fallback: %s", exc)

    return _fallback_embedding(text)


def _fallback_embedding(text: str, dims: int = 256) -> list[float]:
    """Deterministic local embedding used when external service is unavailable."""
    normalized = normalize_payee(text)
    if not normalized:
        return [0.0] * dims

    vector = np.zeros(dims, dtype=np.float32)

    tokens = normalized.split()
    for token in tokens:
        _accumulate_feature(vector, f"tok:{token}")

    compact = normalized.replace(" ", "")
    for size in (3, 4):
        if len(compact) < size:
            continue
        for idx in range(len(compact) - size + 1):
            _accumulate_feature(vector, f"ng:{compact[idx:idx + size]}")

    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


def _accumulate_feature(vector: np.ndarray, feature: str) -> None:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "big") % len(vector)
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    weight = 1.0 + (digest[5] / 255.0)
    vector[bucket] += sign * weight


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def classify(
    user_id: str,
    normalized_payee: str,
    amount_bin: str,
    day_of_week: str,
    day_of_month: int,
) -> Optional[tuple[str, float]]:
    """Attempt Layer 2 classification.

    Returns (category, max_similarity) if a confident match is found,
    or None to fall through to Layer 1.
    """
    examples = await db.get_user_examples(user_id)
    if len(examples) < LAYER2_MIN_EXAMPLES:
        return None

    query_text = (
        f"{normalized_payee} {amount_bin} {day_of_week} {day_of_month}"
    )
    query_vec = await get_embedding(query_text)
    if query_vec is None:
        return None

    scored: list[tuple[float, str]] = []
    for ex in examples:
        stored_vec = ex.get("embedding_vector")
        if not stored_vec:
            continue
        sim = _cosine_similarity(query_vec, stored_vec)
        scored.append((sim, ex["custom_category"]))

    if not scored:
        return None

    scored.sort(reverse=True)
    top_k = scored[: LAYER2_TOP_K]
    max_sim = top_k[0][0]

    if max_sim < LAYER2_SIMILARITY_THRESHOLD:
        return None

    votes = Counter(cat for _, cat in top_k)
    winner, _ = votes.most_common(1)[0]
    return winner, max_sim
