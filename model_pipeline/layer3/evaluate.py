#!/usr/bin/env python3
"""
model_pipeline/layer3/evaluate.py

Evaluation for Layer 3 DBSCAN clustering + LLM naming.

Three things measured:
  1. Cluster quality  — silhouette, coverage, noise %, cluster size
  2. Naming accuracy  — LLM suggested name vs. majority label on pure clusters
  3. Eps sensitivity  — silhouette + coverage at tight / default / loose eps
                        (no LLM calls during sweep)

Usage (from project root):
    python model_pipeline/layer3/evaluate.py
"""

import logging
import os
import sys
import tempfile
from collections import Counter

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_pipeline.layer3.cluster import cluster_user
from model_pipeline.layer3.namer import name_cluster
from model_pipeline.layer2.user_store import load_store_dict

_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "..", "layer2", "config.yaml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dbscan_metrics(user_data: dict, eps: float, min_samples: int) -> dict | None:
    """
    Run DBSCAN and compute unsupervised quality metrics. No LLM calls.
    Returns None when the user has fewer than min_samples transactions.
    """
    embeddings: np.ndarray = user_data["embeddings"]
    n_total = len(user_data["payees"])

    if n_total < min_samples:
        return None

    cluster_labels = DBSCAN(
        eps=eps, min_samples=min_samples, metric="cosine"
    ).fit_predict(embeddings)

    unique_clusters = sorted(set(cluster_labels) - {-1})
    n_clusters      = len(unique_clusters)
    n_noise         = int((cluster_labels == -1).sum())
    coverage        = 1.0 - n_noise / n_total

    sil = float("nan")
    if n_clusters >= 2:
        non_noise = cluster_labels != -1
        sil = float(silhouette_score(
            embeddings[non_noise], cluster_labels[non_noise], metric="cosine"
        ))

    cluster_sizes = [int((cluster_labels == c).sum()) for c in unique_clusters]
    mean_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0

    return {
        "n_clusters":        n_clusters,
        "n_noise":           n_noise,
        "coverage":          coverage,
        "silhouette":        sil,
        "mean_cluster_size": mean_size,
    }


def _eval_naming(
    clusters: list[dict],
    purity_threshold: float,
) -> tuple[int, int, int]:
    """
    For each cluster pure enough to have a ground-truth label, call name_cluster()
    and check if the suggestion matches the majority label.

    Returns (n_matches, n_pure_evaluated, n_skipped).
    """
    n_matches   = 0
    n_evaluated = 0
    n_skipped   = 0

    for cluster in clusters:
        existing = cluster["existing_labels"]
        if not existing:
            continue
        majority_label, majority_count = Counter(existing).most_common(1)[0]
        purity = majority_count / len(existing)
        if purity < purity_threshold:
            n_skipped += 1
            continue
        suggested = name_cluster(cluster["payees"], existing)
        if suggested.strip().lower() == majority_label.strip().lower():
            n_matches += 1
        n_evaluated += 1

    return n_matches, n_evaluated, n_skipped


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_evaluation(config: dict, purity_threshold: float = 0.8) -> None:
    store_path   = config["layer2"]["store_path"]
    eps          = float(config["layer3"]["eps"])
    min_samples  = int(config["layer3"]["min_samples"])
    tracking_uri = config["mlflow"]["tracking_uri"].strip()

    store: dict = load_store_dict(store_path)
    if not store:
        logger.warning("user_store.pkl not found at %s — exiting", store_path)
        return

    logger.info("Loaded store: %d users", len(store))

    # ── Per-user pass: cluster quality + naming accuracy ───────────────────────
    user_rows     = []
    silhouettes   = []
    coverages     = []
    all_mean_sizes= []
    total_txns    = 0
    total_noise   = 0
    total_matches = 0
    total_pure    = 0
    total_skipped = 0
    n_users_eval  = 0

    for user_id, user_data in store.items():
        n_txns = len(user_data.get("payees", []))
        if n_txns < min_samples:
            continue

        n_users_eval += 1
        total_txns   += n_txns

        # ── Cluster quality ────────────────────────────────────────────────────
        m = _dbscan_metrics(user_data, eps, min_samples)
        if m is None:
            continue

        total_noise += m["n_noise"]
        coverages.append(m["coverage"])
        if not np.isnan(m["silhouette"]):
            silhouettes.append(m["silhouette"])
        if m["n_clusters"] > 0:
            all_mean_sizes.append(m["mean_cluster_size"])

        # ── Naming accuracy (LLM calls for pure clusters only) ─────────────────
        clusters = cluster_user(user_data, eps=eps, min_samples=min_samples)
        u_matches, u_pure, u_skipped = _eval_naming(clusters, purity_threshold)

        total_matches += u_matches
        total_pure    += u_pure
        total_skipped += u_skipped

        u_naming_acc = (u_matches / u_pure) if u_pure > 0 else float("nan")

        user_rows.append({
            "user_id":         user_id,
            "n_transactions":  n_txns,
            "n_clusters":      m["n_clusters"],
            "n_noise":         m["n_noise"],
            "coverage":        round(m["coverage"], 4),
            "silhouette":      round(m["silhouette"], 4) if not np.isnan(m["silhouette"]) else float("nan"),
            "n_pure_clusters": u_pure,
            "naming_accuracy": round(u_naming_acc, 4) if not np.isnan(u_naming_acc) else float("nan"),
        })

    # ── Aggregated metrics ─────────────────────────────────────────────────────
    mean_sil   = float(np.mean(silhouettes))    if silhouettes    else float("nan")
    median_sil = float(np.median(silhouettes))  if silhouettes    else float("nan")
    mean_cov   = float(np.mean(coverages))      if coverages      else float("nan")
    mean_cs    = float(np.mean(all_mean_sizes)) if all_mean_sizes else 0.0
    noise_pct  = (total_noise / total_txns * 100) if total_txns   else 0.0
    naming_acc = (total_matches / total_pure)   if total_pure     else float("nan")

    # ── Eps sensitivity sweep (no LLM) ─────────────────────────────────────────
    eps_variants = [
        ("eps_tight",   eps * 0.5),
        ("eps_default", eps),
        ("eps_loose",   eps * 2.0),
    ]
    eps_results: dict[str, dict] = {}

    for tag, eps_val in eps_variants:
        sils, covs = [], []
        for user_data in store.values():
            if len(user_data.get("payees", [])) < min_samples:
                continue
            m = _dbscan_metrics(user_data, eps_val, min_samples)
            if m is None:
                continue
            covs.append(m["coverage"])
            if not np.isnan(m["silhouette"]):
                sils.append(m["silhouette"])
        eps_results[tag] = {
            "eps":       eps_val,
            "silhouette": float(np.mean(sils)) if sils else float("nan"),
            "coverage":   float(np.mean(covs)) if covs else float("nan"),
        }

    # ── Console output ─────────────────────────────────────────────────────────
    _D = "═" * 48
    _d = "─" * 45
    print(f"\n{_D}")
    print(f"  Layer 3 Evaluation")
    print(f"  Users evaluated     : {n_users_eval}")
    print(f"  Mean silhouette     : {mean_sil:.3f}")
    print(f"  Mean coverage       : {mean_cov:.3f}")
    print(f"  Mean cluster size   : {mean_cs:.1f} payees")
    print(f"  Noise %             : {noise_pct:.1f}%")
    print(f"  {_d}")
    print(f"  Naming accuracy     : {naming_acc:.3f}  (on {total_pure} pure clusters)")
    print(f"  Clusters skipped    : {total_skipped}     (purity < {purity_threshold:.2f}, likely new categories)")
    print(f"  {_d}")
    print(f"  Eps sensitivity:")
    for tag, er in eps_results.items():
        label = tag.replace("eps_", "")
        print(
            f"    eps={er['eps']:.3f} {f'({label})':9s}: "
            f"silhouette={er['silhouette']:.3f}  coverage={er['coverage']:.3f}"
        )
    print(f"{_D}\n")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("layer3-evaluation")

    with mlflow.start_run(run_name="layer3-evaluation"):
        # Params
        mlflow.log_param("eps",               eps)
        mlflow.log_param("min_samples",       min_samples)
        mlflow.log_param("purity_threshold",  purity_threshold)
        mlflow.log_param("n_users_evaluated", n_users_eval)
        mlflow.log_param("store_path",        store_path)

        # Core metrics (NaN → 0.0 so MLflow always gets a numeric value)
        def _safe(v: float) -> float:
            return 0.0 if np.isnan(v) else v

        mlflow.log_metric("mean_silhouette",           _safe(mean_sil))
        mlflow.log_metric("median_silhouette",         _safe(median_sil))
        mlflow.log_metric("mean_coverage",             _safe(mean_cov))
        mlflow.log_metric("mean_cluster_size",         mean_cs)
        mlflow.log_metric("n_noise_pct",               noise_pct)
        mlflow.log_metric("naming_accuracy",           _safe(naming_acc))
        mlflow.log_metric("n_pure_clusters_evaluated", float(total_pure))
        mlflow.log_metric("n_clusters_skipped",        float(total_skipped))

        # Eps sensitivity
        for tag, er in eps_results.items():
            mlflow.log_metric(f"silhouette_{tag}", _safe(er["silhouette"]))
            mlflow.log_metric(f"coverage_{tag}",   _safe(er["coverage"]))

        # Per-user artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "layer3_eval_results.csv")
            pd.DataFrame(user_rows).to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            logger.info("Artifact logged: layer3_eval_results.csv (%d rows)", len(user_rows))

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    _config_path = os.path.join(os.path.dirname(__file__), "..", "layer2", "config.yaml")
    with open(_config_path) as _f:
        _config = yaml.safe_load(_f)
    run_evaluation(_config)
