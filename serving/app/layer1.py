"""
Layer 1 — shared Layer-1 classifier.

Supports two loading paths, controlled by ``LAYER1_MODEL_KIND``:

* ``hf``      — download a raw HuggingFace checkpoint logged as MLflow
                artifacts (e.g. the MiniLM run) and run inference with
                ``transformers`` + ``torch``.
* ``pyfunc``  — load an ``mlflow.pyfunc`` packaged model from a model-registry
                URI (e.g. ``models:/layer1-classifier/latest``).

Falls back to a deterministic mock classifier when neither path succeeds.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

from app.config import (
    LABEL_CLASSES,
    LAYER1_HF_MAX_LENGTH,
    LAYER1_MLFLOW_ARTIFACT_PATH,
    LAYER1_MLFLOW_RUN_ID,
    LAYER1_MODEL_KIND,
    MLFLOW_MODEL_URI,
    MLFLOW_TRACKING_URI,
)

logger = logging.getLogger(__name__)

_model: Optional[object] = None
_tokenizer: Optional[object] = None
_model_kind: str = "mock"
_model_version: str = "mock-v0"
_id2label: list[str] = list(LABEL_CLASSES)


def load_model() -> None:
    """Try to load the Layer-1 model; fall back to mock on any failure."""
    global _model, _tokenizer, _model_kind, _model_version, _id2label

    if LAYER1_MODEL_KIND == "hf":
        try:
            _load_hf_from_mlflow()
            return
        except Exception as exc:
            logger.warning("HF load failed (%s) — trying pyfunc", exc)

    try:
        _load_pyfunc()
        return
    except Exception as exc:
        logger.warning("pyfunc load failed (%s) — using mock classifier", exc)

    _model = None
    _tokenizer = None
    _model_kind = "mock"
    _model_version = "mock-v0"


def _load_hf_from_mlflow() -> None:
    """Download raw HF checkpoint from MLflow artifacts and load it."""
    global _model, _tokenizer, _model_kind, _model_version, _id2label

    import mlflow
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    local_dir = mlflow.artifacts.download_artifacts(
        run_id=LAYER1_MLFLOW_RUN_ID,
        artifact_path=LAYER1_MLFLOW_ARTIFACT_PATH,
    )
    logger.info("Downloaded HF artifacts to %s", local_dir)

    _tokenizer = AutoTokenizer.from_pretrained(local_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(local_dir)
    _model.eval()  # type: ignore[union-attr]

    cfg_id2label = getattr(_model.config, "id2label", None)  # type: ignore[union-attr]
    if cfg_id2label and not all(
        str(v).startswith("LABEL_") for v in cfg_id2label.values()
    ):
        _id2label = [cfg_id2label[i] for i in sorted(cfg_id2label, key=lambda k: int(k))]
    else:
        _id2label = list(LABEL_CLASSES)
    logger.info("Layer 1 label order: %s", _id2label[:5])

    _model_kind = "hf"
    _model_version = f"mlflow-run:{LAYER1_MLFLOW_RUN_ID}/{LAYER1_MLFLOW_ARTIFACT_PATH}"
    logger.info("Layer 1 HF model loaded: %s", _model_version)


def _load_pyfunc() -> None:
    """Load an mlflow.pyfunc model from a registry/URI."""
    global _model, _model_kind, _model_version

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
    _model_kind = "pyfunc"
    _model_version = MLFLOW_MODEL_URI
    logger.info("Layer 1 pyfunc model loaded: %s", MLFLOW_MODEL_URI)


def get_model_version() -> str:
    return _model_version


def predict(feature_vector: str) -> tuple[str, float]:
    """Return (predicted_category, confidence)."""
    if _model_kind == "hf":
        return _predict_hf(feature_vector)
    if _model_kind == "pyfunc":
        return _predict_pyfunc(feature_vector)
    return _predict_mock(feature_vector)


def _predict_hf(feature_vector: str) -> tuple[str, float]:
    """Run transformer inference and return (label, softmax_prob)."""
    import torch

    inputs = _tokenizer(  # type: ignore[misc]
        feature_vector,
        truncation=True,
        max_length=LAYER1_HF_MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(**inputs).logits  # type: ignore[union-attr]
    probs = torch.softmax(logits, dim=-1)[0]
    top_idx = int(torch.argmax(probs).item())
    confidence = float(probs[top_idx].item())

    label = _id2label[top_idx] if top_idx < len(_id2label) else "Other"
    if label not in LABEL_CLASSES:
        label = "Other"
    return label, round(confidence, 4)


def _predict_pyfunc(feature_vector: str) -> tuple[str, float]:
    """Run the real MLflow pyfunc model."""
    import pandas as pd

    df = pd.DataFrame({"text": [feature_vector]})
    result = _model.predict(df)  # type: ignore[union-attr]

    if hasattr(result, "iloc"):
        pred_label = str(result.iloc[0])
    elif isinstance(result, list):
        pred_label = str(result[0])
    else:
        pred_label = str(result)

    confidence = 0.75
    if pred_label not in LABEL_CLASSES:
        pred_label = "Other"
        confidence = 0.3

    return pred_label, confidence


def _predict_mock(feature_vector: str) -> tuple[str, float]:
    """Deterministic mock: hash the feature vector to pick a class + confidence."""
    digest = hashlib.md5(feature_vector.encode()).hexdigest()
    idx = int(digest[:8], 16) % len(LABEL_CLASSES)
    conf_raw = int(digest[8:12], 16) / 0xFFFF
    confidence = round(0.40 + conf_raw * 0.55, 4)
    return LABEL_CLASSES[idx], confidence
