"""
models/minilm_finetune.py
─────────────────────────
Fine-tunes sentence-transformers/all-MiniLM-L6-v2 for transaction
categorization. All training logic lives in transformer_base.py.

Config keys read from config["minilm"]:
    learning_rate, num_epochs, batch_size, warmup_steps, max_length
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.layer1.transformer_base import train_transformer, TransformerClassifier

HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
        model_config_key="minilm",
    )