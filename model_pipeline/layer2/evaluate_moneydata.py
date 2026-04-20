#!/usr/bin/env python3
"""
model_pipeline/layer2/evaluate_moneydata.py

Bootstraps Layer 2 user store with moneydata_user 2015-2020 history, then
evaluates the combined Layer 1 + Layer 2 predictor on the 2021-2022 holdout.

Demonstrates that Layer 2 bootstrapping solves the cold-start problem for
out-of-distribution (UK) data where Layer 1 alone scores weighted F1 = 0.1092.

Usage (from project root or layer2 dir):
    python model_pipeline/layer2/evaluate_moneydata.py
    python evaluate_moneydata.py
"""

import sys
import os
import io
import json
import pickle
import tempfile

import numpy as np
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml
import mlflow
from sklearn.metrics import f1_score, classification_report

from model_pipeline.layer2.build_store import make_minio_client
from model_pipeline.layer2.embedder import Embedder
from model_pipeline.evaluate import predict_batch, _load_layer1
from training.utils import normalize_payee

# ── Constants ──────────────────────────────────────────────────────────────────
_MINIO_ENDPOINT  = "http://10.43.4.193:9000"
_MINIO_BUCKET    = "data"
_MINIO_OBJECT    = "raw/evaluation_moneydata.csv"
_CONFIG_PATH     = os.path.join(_SCRIPT_DIR, "config.yaml")
_CUSTOM_STORE_PATH = os.path.join(_PROJECT_ROOT, "artifacts", "moneydata_user_store.pkl")

_BEFORE_WEIGHTED_F1 = 0.1092
_BEFORE_LAYER2_PCT  = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _download_csv(cfg: dict, obj_path: str) -> pd.DataFrame:
    # Temporarily swap the minio object path so make_minio_client can be reused
    client = make_minio_client(cfg)
    response = client.get_object(_MINIO_BUCKET, obj_path)
    df = pd.read_csv(io.BytesIO(response.read()))
    response.close()
    response.release_conn()
    return df


def _build_store_in_memory(df: pd.DataFrame, embedder: Embedder) -> dict:
    """
    Batch-embed all bootstrap payees and build per-user store dict.
    Payees are normalize_payee'd before embedding per spec.
    Embedder.embed_batch already unit-normalizes — no extra step needed.
    Saves once at the end rather than per-row to avoid 4 k+ disk writes.
    """
    payees_normalized = [normalize_payee(p) for p in df["payee"].tolist()]
    print(f"  Embedding {len(payees_normalized):,} transactions (batch) …")
    embeddings = embedder.embed_batch(payees_normalized)  # (n, 768), unit-normalized

    store: dict = {}
    for idx, row in enumerate(df.itertuples(index=False)):
        uid = row.user_id
        if uid not in store:
            store[uid] = {"embeddings": [], "labels": [], "payees": []}
        store[uid]["embeddings"].append(embeddings[idx])
        store[uid]["labels"].append(row.category)
        store[uid]["payees"].append(payees_normalized[idx])

    for uid in store:
        store[uid]["embeddings"] = np.array(store[uid]["embeddings"], dtype=np.float32)

    return store


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    l2 = cfg["layer2"]

    # ── Step 1: Load & split data ──────────────────────────────────────────────
    print("=" * 62)
    print("Step 1 — Downloading evaluation_moneydata.csv from MinIO")
    print("=" * 62)

    df = _download_csv(cfg, _MINIO_OBJECT)
    df["date"]  = pd.to_datetime(df["date"])
    df["_year"] = df["date"].dt.year

    df_bootstrap = df[df["_year"] <= 2020].reset_index(drop=True)
    df_eval      = df[df["_year"] >= 2021].reset_index(drop=True)

    print(f"  Total rows            : {len(df):,}")
    print(f"  Bootstrap (2015-2020) : {len(df_bootstrap):,} rows")
    print(f"  Eval      (2021-2022) : {len(df_eval):,} rows")

    # ── Step 2: Bootstrap user store ───────────────────────────────────────────
    print()
    print("=" * 62)
    print("Step 2 — Building bootstrapped user store")
    print("=" * 62)

    embedder = Embedder(model_name=l2["model_name"], max_length=l2.get("max_length", 128))
    store    = _build_store_in_memory(df_bootstrap, embedder)
    n_added  = sum(len(v["labels"]) for v in store.values())

    os.makedirs(os.path.dirname(_CUSTOM_STORE_PATH), exist_ok=True)
    with open(_CUSTOM_STORE_PATH, "wb") as fh:
        pickle.dump(store, fh)

    print(f"  {n_added:,} transactions added")
    print(f"  Store saved → {_CUSTOM_STORE_PATH}")

    # ── Step 3: Evaluate on 2021-2022 holdout ─────────────────────────────────
    print()
    print("=" * 62)
    print("Step 3 — Evaluating Layer 1 + Layer 2 on 2021-2022 holdout")
    print("=" * 62)

    print("  Loading Layer 1 model from MLflow …")
    layer1_model = _load_layer1(cfg)

    # Drop holdout rows whose categories fall outside the Layer 1 label set
    known_labels = {
        lbl.replace("__label__", "").replace("_", " ")
        for lbl in layer1_model.get_labels()
    }
    before_drop = len(df_eval)
    df_eval = df_eval[df_eval["category"].isin(known_labels)].reset_index(drop=True)
    dropped = before_drop - len(df_eval)
    if dropped:
        print(f"  Dropped {dropped} rows with unknown categories. Remaining: {len(df_eval):,}")

    # predict_batch from evaluate.py: no store mutation, pure batch inference
    results = predict_batch(
        df_eval, layer1_model, embedder, store,
        k=l2["k"],
        threshold=l2["similarity_threshold"],
        min_history=l2["min_history"],
    )

    y_true = results["true_label"].tolist()
    y_pred = results["pred_label"].tolist()

    layer2_count = int((results["source"] == "layer2").sum())
    layer2_pct   = layer2_count / len(results) * 100
    weighted_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1     = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    report_dict  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"\n  Layer 2 routing : {layer2_count:,}/{len(results):,}  ({layer2_pct:.1f}%)")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}")

    # ── Step 4: Log to MLflow ──────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("Step 4 — Logging to MLflow")
    print("=" * 62)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].strip())
    mlflow.set_experiment("layer1-layer2-evaluation")

    with mlflow.start_run(run_name="fasttext-layer2-moneydata-bootstrapped"):
        mlflow.log_metrics({
            "weighted_f1":           weighted_f1,
            "macro_f1":              macro_f1,
            "layer2_routing_pct":    layer2_pct,
            "bootstrap_transactions": float(n_added),
            "eval_transactions":     float(len(df_eval)),
        })
        mlflow.set_tags({
            "dataset":  "evaluation_moneydata",
            "approach": "layer2_bootstrapped",
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "classification_report_moneydata.json")
            with open(report_path, "w") as fh:
                json.dump(report_dict, fh, indent=2)
            mlflow.log_artifact(report_path)

            impact = {
                "before_weighted_f1": _BEFORE_WEIGHTED_F1,
                "after_weighted_f1":  round(weighted_f1, 4),
                "before_layer2_pct":  _BEFORE_LAYER2_PCT,
                "after_layer2_pct":   round(layer2_pct, 2),
            }
            impact_path = os.path.join(tmpdir, "bootstrapping_impact.json")
            with open(impact_path, "w") as fh:
                json.dump(impact, fh, indent=2)
            mlflow.log_artifact(impact_path)

    print("  Metrics and artifacts logged.")

    # ── Step 5: Before / after summary ────────────────────────────────────────
    print()
    print("=" * 62)
    print("  BEFORE / AFTER BOOTSTRAPPING — moneydata_user")
    print("=" * 62)
    print(f"  {'Metric':<28} {'Before':>9} {'After':>9}")
    print("  " + "-" * 48)
    print(f"  {'Weighted F1':<28} {_BEFORE_WEIGHTED_F1:>9.4f} {weighted_f1:>9.4f}")
    print(f"  {'Macro F1':<28} {'N/A':>9} {macro_f1:>9.4f}")
    print(f"  {'Layer 2 routing %':<28} {_BEFORE_LAYER2_PCT:>9.1f} {layer2_pct:>9.1f}")
    print(f"  {'Bootstrap transactions':<28} {'0':>9} {n_added:>9,}")
    print(f"  {'Eval transactions':<28} {'':>9} {len(df_eval):>9,}")
    print("=" * 62)
    print(f"\n  Weighted F1 improvement : {weighted_f1 - _BEFORE_WEIGHTED_F1:+.4f}")
    print(f"  Layer 2 routing shift   : {_BEFORE_LAYER2_PCT:.1f}% → {layer2_pct:.1f}%")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
