"""
layer2/user_store.py

Manages the per-user vector store at runtime.

Store structure:
    {
        user_id: {
            "embeddings": np.ndarray  # (n, 768)
            "labels":     list[str]   # category strings, parallel to embeddings
            "payees":     list[str]   # original payee strings, for interpretability
        }
    }

All embeddings must be unit-normalized before being passed to add_transaction.
"""

import os
import pickle
import numpy as np
from typing import Optional


_store: dict = {}
_store_path: str = ""


def load_store(path: str) -> dict:
    """Load user_store.pkl from disk into memory. Returns the store dict."""
    global _store, _store_path
    _store_path = path
    if os.path.exists(path):
        with open(path, "rb") as f:
            _store = pickle.load(f)
    else:
        _store = {}
    return _store


def _save_store() -> None:
    """Persist current in-memory store to disk."""
    if not _store_path:
        raise RuntimeError("Store path not set. Call load_store() first.")
    tmp_path = _store_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(_store, f)
    os.replace(tmp_path, _store_path)  # atomic rename


def get_user_store(user_id: str) -> Optional[dict]:
    """Return a single user's store dict, or None if user not found."""
    return _store.get(user_id, None)


def add_transaction(
    user_id: str,
    payee: str,
    embedding: np.ndarray,
    category: str,
) -> None:
    """
    Append a new transaction to the user's in-memory store and persist to disk.

    Args:
        user_id:   user identifier
        payee:     original payee string
        embedding: (768,) unit-normalized embedding
        category:  ground-truth or predicted category label
    """
    if user_id not in _store:
        _store[user_id] = {
            "embeddings": np.empty((0, embedding.shape[0]), dtype=np.float32),
            "labels": [],
            "payees": [],
        }

    user_data = _store[user_id]
    user_data["embeddings"] = np.vstack(
        [user_data["embeddings"], embedding.reshape(1, -1).astype(np.float32)]
    )
    user_data["labels"].append(category)
    user_data["payees"].append(payee)

    _save_store()


def has_sufficient_history(user_id: str, min_history: int) -> bool:
    """Return True if user has at least min_history stored transactions."""
    user_data = _store.get(user_id)
    if user_data is None:
        return False
    return len(user_data["labels"]) >= min_history