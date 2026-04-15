"""
models/mpnet_finetune.py
────────────────────────
Fine-tunes sentence-transformers/all-mpnet-base-v2 for transaction
categorization. All training logic lives in transformer_base.py.

Difference from MiniLM/DistilBERT: uses mean pooling over all token
embeddings (with attention-mask weighting) instead of the CLS token.
all-mpnet was pretrained with mean pooling — using CLS works but leaves
generalization performance on the table.

Config keys read from config["mpnet"]:
    learning_rate, num_epochs, batch_size, warmup_steps, max_length
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.layer1.transformer_base import train_transformer, TransformerClassifier

HF_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


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
        model_config_key="mpnet",
        pooling="mean",          # all-mpnet requires mean pooling
    )