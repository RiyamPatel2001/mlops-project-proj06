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

import io
import os
import pickle
from urllib.parse import urlparse
import numpy as np
from typing import Optional


_store: dict = {}
_store_path: str = ""
_minio_url: str = ""


def _parse_minio_url(url: str):
    p = urlparse(url)
    endpoint = f"{p.scheme}://{p.netloc}"
    bucket, obj = p.path.lstrip("/").split("/", 1)
    return endpoint, bucket, obj


def _minio_client(endpoint: str):
    from minio import Minio
    ep = endpoint.replace("http://", "").replace("https://", "")
    return Minio(
        ep,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=endpoint.startswith("https://"),
    )


def load_store_dict(path: str) -> dict:
    """Load user_store.pkl from disk or MinIO. Returns the store dict without touching globals."""
    if path.startswith("http://") or path.startswith("https://"):
        endpoint, bucket, obj = _parse_minio_url(path)
        try:
            client = _minio_client(endpoint)
            response = client.get_object(bucket, obj)
            store = pickle.loads(response.read())
            response.close()
            response.release_conn()
            return store
        except Exception:
            return {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def load_store(path: str) -> dict:
    """Load user_store.pkl from disk or MinIO into memory. Returns the store dict."""
    global _store, _store_path, _minio_url
    if path.startswith("http://") or path.startswith("https://"):
        _minio_url = path
        endpoint, bucket, obj = _parse_minio_url(path)
        _store_path = f"/tmp/{obj.replace('/', '_')}"
    else:
        _minio_url = ""
        _store_path = path
    _store = load_store_dict(path)
    return _store


def _save_store() -> None:
    """Persist current in-memory store to local file and upload to MinIO if applicable."""
    if not _store_path:
        raise RuntimeError("Store path not set. Call load_store() first.")
    data = pickle.dumps(_store)
    tmp_path = _store_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, _store_path)
    if _minio_url:
        endpoint, bucket, obj = _parse_minio_url(_minio_url)
        _minio_client(endpoint).put_object(bucket, obj, io.BytesIO(data), len(data))


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