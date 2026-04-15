"""
model_pipeline/layer2/evaluate.py

Unified inference interface. Loads Layer 1 (fastText) and Layer 2 components
at startup. Exposes a single predict() function that returns the agreed output schema.

Usage:
    from model_pipeline.layer2.predictor import Predictor
    predictor = Predictor(config_path="config.yaml")
    result = predictor.predict(
        transaction_id="txn_001",
        user_id="user_42",
        payee="Whole Foods Market",
        amount=54.20,
        date="2024-03-15",
    )
"""

import os
import yaml
import mlflow
import fasttext
import numpy as np

from model_pipeline.layer2.embedder import Embedder
from model_pipeline.layer2 import user_store as store_module
from model_pipeline.layer2.matcher import get_top_k, majority_vote


class Predictor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        l2 = cfg["layer2"]

        self.k = l2["k"]
        self.threshold = l2["similarity_threshold"]
        self.min_history = l2["min_history"]
        self.store_path = l2["store_path"]

        # --- Load Layer 1 (fastText) ---
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        layer1_model_uri = cfg["layer1"]["model_uri"]
        local_path = mlflow.artifacts.download_artifacts(layer1_model_uri)
        # fastText artifact is a .bin file inside the downloaded directory
        bin_files = [f for f in os.listdir(local_path) if f.endswith(".bin")]
        if not bin_files:
            raise FileNotFoundError(f"No .bin file found in {local_path}")
        self.layer1_model = fasttext.load_model(os.path.join(local_path, bin_files[0]))
        self.model_version = cfg["layer1"].get("model_version", "fasttext-v1")

        # --- Load Layer 2 ---
        self.embedder = Embedder(
            model_name=l2["model_name"],
            max_length=l2.get("max_length", 128),
        )
        store_module.load_store(self.store_path)

    def _layer1_predict(self, payee: str):
        """Returns (category, confidence) from fastText."""
        labels, probs = self.layer1_model.predict(payee, k=1)
        category = labels[0].replace("__label__", "")
        confidence = float(probs[0])
        return category, confidence

    def predict(
        self,
        transaction_id: str,
        user_id: str,
        payee: str,
        amount: float,
        date: str,
    ) -> dict:
        """
        Predict category for a transaction. Combines Layer 1 and Layer 2.

        Returns dict with keys:
            transaction_id, prediction_category, confidence, model_version, source
        """
        # Step 1: Layer 1 prediction (always computed)
        l1_category, l1_confidence = self._layer1_predict(payee)

        # Step 2: Check cold-start condition
        if not store_module.has_sufficient_history(user_id, self.min_history):
            result = {
                "transaction_id": transaction_id,
                "prediction_category": l1_category,
                "confidence": l1_confidence,
                "model_version": self.model_version,
                "source": "layer1",
            }
            # Still add to store to accumulate history
            embedding = self.embedder.embed(payee)
            store_module.add_transaction(user_id, payee, embedding, l1_category)
            return result

        # Step 3: Embed query
        query_embedding = self.embedder.embed(payee)

        # Step 4 & 5: Get top-k neighbors and majority vote
        user_data = store_module.get_user_store(user_id)
        neighbors = get_top_k(query_embedding, user_data, self.k)
        l2_category, l2_confidence, threshold_exceeded = majority_vote(neighbors, self.threshold)

        # Step 6 & 7: Decide which prediction to return
        if threshold_exceeded:
            result = {
                "transaction_id": transaction_id,
                "prediction_category": l2_category,
                "confidence": l2_confidence,
                "model_version": self.model_version,
                "source": "layer2",
            }
        else:
            result = {
                "transaction_id": transaction_id,
                "prediction_category": l1_category,
                "confidence": l1_confidence,
                "model_version": self.model_version,
                "source": "layer1",
            }

        # Step 8: Update store synchronously (use ground truth label if available,
        # otherwise use the prediction we're returning)
        store_module.add_transaction(
            user_id, payee, query_embedding, result["prediction_category"]
        )

        return result