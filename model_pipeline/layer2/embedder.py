"""
layer2/embedder.py

Loads sentence-transformers/all-mpnet-base-v2 and exposes two functions:
  embed(payee_string) -> np.array (768,)
  embed_batch(payee_strings) -> np.array (n, 768)

All embeddings are unit-normalized so cosine similarity reduces to a dot product.
Instantiate Embedder once at startup and reuse across calls.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", max_length: int = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pool(self, model_output, attention_mask) -> torch.Tensor:
        """Mean pooling over token embeddings with attention mask applied."""
        token_embeddings = model_output.last_hidden_state  # (batch, seq_len, 768)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask  # (batch, 768)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows so cosine similarity == dot product."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def embed_batch(self, payee_strings: list, batch_size: int = 256) -> np.ndarray:
        """Embed a list of payee strings. Returns (n, 768) float32 array, unit-normalized."""
        all_embeddings = []
        for i in range(0, len(payee_strings), batch_size):
            batch = payee_strings[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            pooled = self._mean_pool(output, encoded["attention_mask"])
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
        embeddings = np.vstack(all_embeddings)
        return self._normalize(embeddings)

    def embed(self, payee_string: str) -> np.ndarray:
        """Embed a single payee string. Returns (768,) float32 array, unit-normalized."""
        return self.embed_batch([payee_string])[0]