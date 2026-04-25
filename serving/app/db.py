"""Async Postgres connection pool and queries for serving data + auth records."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import asyncpg

from app.config import POSTGRES_DSN

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


# ── Pool lifecycle ───────────────────────────────────────────────────────────

async def init_pool() -> None:
    global _pool
    try:
        _pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=2, max_size=10)
        logger.info("Postgres pool created (%s)", POSTGRES_DSN.split("@")[-1])
    except Exception:
        logger.warning("Could not connect to Postgres — running in mock-db mode")
        _pool = None


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def pool_available() -> bool:
    return _pool is not None


# ── Schema bootstrap ─────────────────────────────────────────────────────────

_AUTH_USERS_DDL = """
CREATE TABLE IF NOT EXISTS auth_users (
    user_id              VARCHAR(36) PRIMARY KEY,
    username             VARCHAR(100) NOT NULL,
    username_normalized  VARCHAR(100) NOT NULL UNIQUE,
    password_hash        TEXT NOT NULL,
    created_at           TIMESTAMP DEFAULT now()
);
"""

_AUTH_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS auth_sessions (
    id          SERIAL PRIMARY KEY,
    user_id     VARCHAR(36) NOT NULL REFERENCES auth_users(user_id) ON DELETE CASCADE,
    token_hash  VARCHAR(64) NOT NULL UNIQUE,
    created_at  TIMESTAMP DEFAULT now(),
    expires_at  TIMESTAMP NOT NULL
);
"""

_AUTH_SESSIONS_USER_IDX_DDL = """
CREATE INDEX IF NOT EXISTS auth_sessions_user_id_idx
    ON auth_sessions(user_id);
"""

_FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS feedback_store (
    id                  SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(50),
    user_id             VARCHAR(50),
    payee               VARCHAR(255),
    amount              INTEGER,
    date                DATE,
    original_prediction VARCHAR(100),
    original_confidence FLOAT,
    source              VARCHAR(20),
    final_label         VARCHAR(100),
    reviewed_by_user    BOOLEAN,
    timestamp           TIMESTAMP DEFAULT now()
);
"""

_LAYER2_DDL = """
CREATE TABLE IF NOT EXISTS layer2_examples (
    id                SERIAL PRIMARY KEY,
    user_id           VARCHAR(50),
    payee             VARCHAR(255),
    custom_category   VARCHAR(100),
    embedding_vector  FLOAT[],
    timestamp         TIMESTAMP DEFAULT now()
);
"""

_SUGGESTION_DDL = """
CREATE TABLE IF NOT EXISTS suggestion_responses (
    id                  SERIAL PRIMARY KEY,
    user_id             VARCHAR(50),
    transaction_id      VARCHAR(50),
    action              VARCHAR(20),
    suggested_category  VARCHAR(100),
    timestamp           TIMESTAMP DEFAULT now()
);
"""

_LAYER3_SUGGESTIONS_DDL = """
CREATE TABLE IF NOT EXISTS layer3_suggestions (
    id                      SERIAL PRIMARY KEY,
    user_id                 TEXT NOT NULL,
    cluster_id              TEXT NOT NULL UNIQUE,
    suggested_category_name TEXT NOT NULL,
    payee_list              TEXT[] NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_LAYER3_SUGGESTIONS_USER_STATUS_IDX_DDL = """
CREATE INDEX IF NOT EXISTS idx_layer3_suggestions_user_status
    ON layer3_suggestions (user_id, status);
"""


async def ensure_tables() -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(_AUTH_USERS_DDL)
        await conn.execute(_AUTH_SESSIONS_DDL)
        await conn.execute(_AUTH_SESSIONS_USER_IDX_DDL)
        await conn.execute(_FEEDBACK_DDL)
        await conn.execute(_LAYER2_DDL)
        await conn.execute(_SUGGESTION_DDL)
        await conn.execute(_LAYER3_SUGGESTIONS_DDL)
        await conn.execute(_LAYER3_SUGGESTIONS_USER_STATUS_IDX_DDL)
    logger.info("Database tables ensured")


# ── Auth CRUD ────────────────────────────────────────────────────────────────

async def create_auth_user(
    user_id: str,
    username: str,
    username_normalized: str,
    password_hash: str,
) -> dict[str, Any] | None:
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        try:
            rec = await conn.fetchrow(
                """
                INSERT INTO auth_users
                    (user_id, username, username_normalized, password_hash)
                VALUES ($1, $2, $3, $4)
                RETURNING user_id, username
                """,
                user_id,
                username,
                username_normalized,
                password_hash,
            )
        except asyncpg.exceptions.UniqueViolationError:
            return None

        return dict(rec) if rec else None


async def get_auth_user_by_username(
    username_normalized: str,
) -> dict[str, Any] | None:
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        rec = await conn.fetchrow(
            """
            SELECT user_id, username, username_normalized, password_hash
            FROM auth_users
            WHERE username_normalized = $1
            """,
            username_normalized,
        )
        return dict(rec) if rec else None


async def create_auth_session(
    user_id: str,
    token_hash: str,
    expires_at: datetime,
) -> None:
    if not _pool:
        return

    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO auth_sessions (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
            """,
            user_id,
            token_hash,
            expires_at,
        )


async def get_auth_user_for_session(
    token_hash: str,
) -> dict[str, Any] | None:
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        rec = await conn.fetchrow(
            """
            SELECT auth_users.user_id, auth_users.username, auth_sessions.expires_at
            FROM auth_sessions
            JOIN auth_users ON auth_users.user_id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = $1
            """,
            token_hash,
        )
        if not rec:
            return None

        expires_at = rec["expires_at"]
        if expires_at is not None and expires_at <= datetime.utcnow():
            await conn.execute(
                "DELETE FROM auth_sessions WHERE token_hash = $1",
                token_hash,
            )
            return None

        return {
            "user_id": rec["user_id"],
            "username": rec["username"],
        }


async def delete_auth_session(token_hash: str) -> None:
    if not _pool:
        return

    async with _pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM auth_sessions WHERE token_hash = $1",
            token_hash,
        )


# ── Feedback CRUD ────────────────────────────────────────────────────────────

async def insert_feedback(row: dict[str, Any]) -> int:
    if not _pool:
        return 0
    async with _pool.acquire() as conn:
        rec = await conn.fetchrow(
            """
            INSERT INTO feedback_store
                (transaction_id, user_id, payee, amount, date,
                 original_prediction, original_confidence, source,
                 final_label, reviewed_by_user, timestamp)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            RETURNING id
            """,
            row["transaction_id"],
            row["user_id"],
            row["payee"],
            row["amount"],
            row["date"],
            row.get("original_prediction"),
            row.get("original_confidence"),
            row.get("source", "layer1"),
            row["final_label"],
            row["reviewed_by_user"],
            row.get("timestamp", datetime.utcnow()),
        )
        return rec["id"]  # type: ignore[index]


async def export_feedback() -> list[dict[str, Any]]:
    """Return reviewed layer-1 feedback rows for Saketh's batch pipeline."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT transaction_id, user_id, payee, amount,
                   date::text AS date, original_prediction, original_confidence,
                   source, final_label, reviewed_by_user, timestamp::text AS timestamp
            FROM feedback_store
            WHERE reviewed_by_user = TRUE AND source = 'layer1'
            ORDER BY timestamp
            """
        )
        return [dict(r) for r in rows]


# ── Layer 2 examples CRUD ────────────────────────────────────────────────────

async def insert_layer2_example(
    user_id: str,
    payee: str,
    custom_category: str,
    embedding: list[float],
) -> int:
    if not _pool:
        return 0
    async with _pool.acquire() as conn:
        rec = await conn.fetchrow(
            """
            INSERT INTO layer2_examples (user_id, payee, custom_category, embedding_vector)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            user_id,
            payee,
            custom_category,
            embedding,
        )
        return rec["id"]  # type: ignore[index]


async def get_user_examples(user_id: str) -> list[dict[str, Any]]:
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, payee, custom_category, embedding_vector
            FROM layer2_examples
            WHERE user_id = $1
            ORDER BY timestamp DESC
            """,
            user_id,
        )
        return [dict(r) for r in rows]


async def get_user_custom_categories(user_id: str) -> list[str]:
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT custom_category
            FROM layer2_examples
            WHERE user_id = $1
            ORDER BY custom_category
            """,
            user_id,
        )
        return [r["custom_category"] for r in rows]


# ── Suggestion responses ─────────────────────────────────────────────────────

async def insert_suggestion_response(
    user_id: str,
    transaction_id: str,
    action: str,
    suggested_category: str,
) -> int:
    if not _pool:
        return 0
    async with _pool.acquire() as conn:
        rec = await conn.fetchrow(
            """
            INSERT INTO suggestion_responses
                (user_id, transaction_id, action, suggested_category)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            user_id,
            transaction_id,
            action,
            suggested_category,
        )
        return rec["id"]  # type: ignore[index]
