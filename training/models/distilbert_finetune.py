"""
models/distilbert_finetune.py
─────────────────────────────
Fine-tunes distilbert-base-uncased for transaction categorization.
All training logic lives in transformer_base.py.

Config keys read from config["distilbert"]:
    learning_rate, num_epochs, batch_size, warmup_steps, max_length

Note: batch_size=16 (vs MiniLM's 32) — DistilBERT is heavier and
will OOM on the RTX 6000 at batch_size=32 with max_length=64.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.transformer_base import train_transformer, TransformerClassifier

HF_MODEL_NAME = "distilbert-base-uncased"


def train(
    X_train: pd.Series | list[str],
    y_train: np.ndarray,
    config: dict,
) -> tuple[None, TransformerClassifier]:
    return train_transformer(
        X_train=X_train,
        y_train=y_train,
        config=config,
        hf_model_name=HF_MODEL_NAME,
        model_config_key="distilbert",
    )