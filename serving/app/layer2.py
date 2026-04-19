"""
Layer 2 — personal history KNN override.

For a given user + payee string:
1. Call Saketh's external embedding service to get the MiniLM vector.
2. Retrieve the user's stored examples from Postgres.
3. Compute cosine similarity between the incoming vector and each stored vector.
4. If the top match >= threshold, do a majority vote among top-k neighbours.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Optional

import httpx

from app import db
from app.config import (
    EMBEDDING_SERVICE_URL,
    LAYER2_MIN_EXAMPLES,
    LAYER2_SIMILARITY_THRESHOLD,
    LAYER2_TOP_K,
)

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
            return data.get("embedding") or data.get("vector")
    except Exception as exc:
        logger.warning("Embedding service unavailable: %s", exc)
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def classify(
    user_id: str,
    payee: str,
) -> Optional[tuple[str, float]]:
    """Attempt Layer 2 classification.

    Returns (category, max_similarity) if a confident match is found,
    or None to fall through to Layer 1.
    """
    examples = await db.get_user_examples(user_id)
    if len(examples) < LAYER2_MIN_EXAMPLES:
        return None

    query_vec = await get_embedding(payee)
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
