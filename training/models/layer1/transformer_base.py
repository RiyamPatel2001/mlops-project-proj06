"""
models/transformer_base.py
──────────────────────────
Shared fine-tuning logic for all HuggingFace transformer candidates.

Both MiniLM and DistilBERT call train_transformer() with their own
model name — everything else (dataset, loop, optimizer, scheduler) is
identical. No duplication between candidates.

Returns a TransformerClassifier wrapper with a .predict() method so
evaluate_and_log() works without modification, same as tfidf and fasttext.

Fix (2025-03): num_workers=2 with on-the-fly HuggingFace tokenization causes
a deadlock after epoch 1 on Linux GPU instances (PyTorch multiprocessing +
HuggingFace tokenizer incompatibility). Fixed by:
  1. Pre-tokenizing the entire training set once into a TensorDataset —
     no per-sample work left for workers to do.
  2. num_workers=0 as a belt-and-suspenders safety measure (also avoids
     any fork-safety issues with the tokenizer's Rust backend).
"""

from __future__ import annotations

import os
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# ── sklearn-compatible wrapper ────────────────────────────────────────────────

class TransformerClassifier:
    """
    Wraps a fine-tuned HuggingFace model with a sklearn-compatible .predict()
    so evaluate_and_log() needs no changes.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int,
        device: torch.device,
        pooling: str = "cls",   # "cls" | "mean"
    ):
        self.model      = model
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.device     = device
        self.pooling    = pooling

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool over non-padding token embeddings.
        Masks padding positions before averaging so they don't dilute the result.
        """
        # Expand mask to match hidden-state dimensions: (batch, seq, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed   = torch.sum(last_hidden_state * mask_expanded, dim=1)
        # Clamp to avoid division by zero on degenerate inputs
        counts   = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return summed / counts

    def predict(self, X: pd.Series | list[str]) -> np.ndarray:
        if isinstance(X, pd.Series):
            X = X.tolist()

        self.model.eval()
        preds = []

        # Batch inference — avoids OOM on large val sets
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**enc, output_hidden_states=(self.pooling == "mean"))

                if self.pooling == "mean":
                    # Use last hidden state with mean pooling, then project
                    # through the model's classifier head manually.
                    # Note: this calls self.model.classifier(pooled) directly,
                    # which is correct for MPNet's AutoModelForSequenceClassification
                    # (single linear head). DistilBERT has an additional pre_classifier
                    # layer — but since only mpnet_finetune.py passes pooling="mean",
                    # this branch is never reached for DistilBERT.
                    last_hidden = outputs.hidden_states[-1]
                    pooled = self._mean_pool(last_hidden, enc["attention_mask"])
                    logits = self.model.classifier(pooled)
                else:
                    logits = outputs.logits

                preds.extend(logits.argmax(dim=-1).cpu().numpy())

        return np.array(preds)

    def save(self, output_dir: str) -> None:
        """Save model + tokenizer in HuggingFace format."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


# ── Core training function ────────────────────────────────────────────────────

def train_transformer(
    X_train: pd.Series | list[str],
    y_train: np.ndarray,
    config: dict,
    hf_model_name: str,
    model_config_key: str,
    pooling: str = "cls",   # "cls" | "mean"
) -> tuple[None, TransformerClassifier]:
    """
    Fine-tune a HuggingFace sequence classification model.

    Args:
        X_train:          payee_norm strings
        y_train:          integer-encoded category labels
        config:           full config dict
        hf_model_name:    HuggingFace model hub name
                            e.g. "sentence-transformers/all-MiniLM-L6-v2"
        model_config_key: key in config for hyperparams
                            e.g. "minilm" or "distilbert"
        pooling:          "cls"  — use CLS token logits (default, DistilBERT/MiniLM)
                          "mean" — mean-pool last hidden state before classifier head
                                   (required for all-mpnet-base-v2 which was pretrained
                                   with mean pooling; CLS works but underperforms)

    Returns:
        (None, TransformerClassifier) — None in the vectorizer slot;
        evaluate_and_log() already handles vec=None.
    """
    model_cfg  = config[model_config_key]
    num_labels = len(set(y_train.tolist()))
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transformer] Device: {device}")

    # Log GPU environment as MLflow params so runs are reproducible
    if torch.cuda.is_available():
        mlflow.log_param("gpu_name",       torch.cuda.get_device_name(0))
        mlflow.log_param("cuda_version",   torch.version.cuda)
        mlflow.log_param("gpu_memory_gb",  round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        ))

    print(f"[transformer] Loading {hf_model_name} ...")

    # ── Load tokenizer + model ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # replaces pretrained head with new one
    ).to(device)

    # ── Pre-tokenize entire training set once ─────────────────────────────────
    # Doing this up front (rather than on-the-fly in __getitem__) avoids the
    # PyTorch multiprocessing + HuggingFace tokenizer deadlock that causes a
    # hang after epoch 1 on Linux GPU instances. It also speeds up subsequent
    # epochs since there's no repeated tokenizer work.
    if isinstance(X_train, pd.Series):
        X_train = X_train.tolist()

    print(f"[transformer] Pre-tokenizing {len(X_train):,} training samples ...")
    encodings = tokenizer(
        X_train,
        max_length=model_cfg["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(y_train, dtype=torch.long),
    )
    print(f"[transformer] Pre-tokenization complete.")

    # num_workers=0: keeps data loading in the main process.
    # Belt-and-suspenders alongside pre-tokenization — eliminates any
    # remaining fork-safety risk from the tokenizer's Rust backend.
    loader = DataLoader(
        dataset,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
    )
    total_steps = len(loader) * model_cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=model_cfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(model_cfg["num_epochs"]):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()

            if step % 50 == 0:
                print(
                    f"  epoch {epoch+1}/{model_cfg['num_epochs']}  "
                    f"step {step}/{len(loader)}  "
                    f"loss={outputs.loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        elapsed  = time.perf_counter() - epoch_start
        print(f"  epoch {epoch+1} complete — avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")

        mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
        mlflow.log_metric("epoch_time", round(elapsed, 2), step=epoch)

    clf = TransformerClassifier(
        model=model,
        tokenizer=tokenizer,
        max_length=model_cfg["max_length"],
        device=device,
        pooling=pooling,
    )
    return None, clf