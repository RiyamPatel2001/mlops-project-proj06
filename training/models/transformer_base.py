"""
models/transformer_base.py
──────────────────────────
Shared fine-tuning logic for all HuggingFace transformer candidates.

Both MiniLM and DistilBERT call train_transformer() with their own
model name — everything else (dataset, loop, optimizer, scheduler) is
identical. No duplication between candidates.

Returns a TransformerClassifier wrapper with a .predict() method so
evaluate_and_log() works without modification, same as tfidf and fasttext.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PayeeDataset(Dataset):
    """
    Tokenizes payee strings on-the-fly.
    Keeps memory footprint small — no pre-tokenizing the full dataset.
    """

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── sklearn-compatible wrapper ────────────────────────────────────────────────

class TransformerClassifier:
    """
    Wraps a fine-tuned HuggingFace model with a sklearn-compatible .predict()
    so evaluate_and_log() needs no changes.
    """

    def __init__(self, model, tokenizer, max_length: int, device: torch.device):
        self.model      = model
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.device     = device

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
                logits = self.model(**enc).logits
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

    Returns:
        (None, TransformerClassifier) — None in the vectorizer slot;
        evaluate_and_log() already handles vec=None.
    """
    model_cfg   = config[model_config_key]
    num_labels  = len(set(y_train.tolist()))
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transformer] Device: {device}")
    print(f"[transformer] Loading {hf_model_name} ...")

    # ── Load tokenizer + model ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # replaces pretrained head with new one
    ).to(device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    if isinstance(X_train, pd.Series):
        X_train = X_train.tolist()

    dataset = PayeeDataset(
        texts=X_train,
        labels=y_train,
        tokenizer=tokenizer,
        max_length=model_cfg["max_length"],
    )
    loader = DataLoader(
        dataset,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
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
        total_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

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
        print(f"  epoch {epoch+1} complete — avg_loss={avg_loss:.4f}")

    clf = TransformerClassifier(
        model=model,
        tokenizer=tokenizer,
        max_length=model_cfg["max_length"],
        device=device,
    )
    return None, clf