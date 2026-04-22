#!/usr/bin/env python3
"""
model_pipeline/layer2/evaluate_moneydata_sliding.py

Sliding-window evaluation of the Layer 1 + Layer 2 pipeline on MoneyData.

Simulates a real user onboarding from scratch: at each eval_year, all
transactions before that year seed the Layer 2 store; the eval_year's
transactions form the test set.  This shows how pipeline performance
improves as personal history accumulates.

Window design:
  eval_year=2015: bootstrap=2014,       test=2015
  eval_year=2016: bootstrap=2014–2015,  test=2016
  ...
  eval_year=2022: bootstrap=2014–2021,  test=2022

Usage (from project root):
    python model_pipeline/layer2/evaluate_moneydata_sliding.py
"""

import os
import sys
import tempfile

import pandas as pd
import yaml
import mlflow
from sklearn.metrics import f1_score

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_pipeline.layer2.evaluate_moneydata import _download_csv, _build_store_in_memory
from model_pipeline.layer2.embedder import Embedder
from model_pipeline.evaluate import predict_batch, _load_layer1

# ── Constants ──────────────────────────────────────────────────────────────────
_CONFIG_PATH  = os.path.join(_SCRIPT_DIR, "config.yaml")
_MINIO_OBJECT = "raw/evaluation_moneydata.csv"
_DIVIDER      = "═" * 48
EVAL_YEARS    = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    l2 = cfg["layer2"]

    # ── Download data ──────────────────────────────────────────────────────────
    print("Downloading evaluation_moneydata.csv from MinIO ...")
    df = _download_csv(cfg, _MINIO_OBJECT)
    df["date"]  = pd.to_datetime(df["date"])
    df["_year"] = df["date"].dt.year
    print(f"  Total rows: {len(df):,}  years: {sorted(df['_year'].unique())}")

    # ── Load shared artifacts (once) ───────────────────────────────────────────
    print("\nLoading Layer 1 model from MLflow ...")
    layer1_model = _load_layer1(cfg)

    known_labels = {
        lbl.replace("__label__", "").replace("_", " ")
        for lbl in layer1_model.get_labels()
    }

    print(f"Loading embedder: {l2['model_name']} ...")
    embedder = Embedder(model_name=l2["model_name"], max_length=l2.get("max_length", 128))

    # ── MLflow run ─────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].strip())
    mlflow.set_experiment("layer1-layer2-evaluation")

    result_rows = []

    with mlflow.start_run(run_name="fasttext-layer2-moneydata-sliding-window"):
        mlflow.set_tag("dataset",  "evaluation_moneydata")
        mlflow.set_tag("approach", "sliding_window")

        for eval_year in EVAL_YEARS:
            df_bootstrap = df[df["_year"] <  eval_year].reset_index(drop=True)
            df_eval      = df[df["_year"] == eval_year].reset_index(drop=True)

            if len(df_eval) == 0:
                print(f"[warning] eval_year={eval_year}: no eval rows — skipping.")
                continue

            # Drop eval rows whose categories fall outside the Layer 1 label set
            df_eval = df_eval[df_eval["category"].isin(known_labels)].reset_index(drop=True)
            if len(df_eval) == 0:
                print(f"[warning] eval_year={eval_year}: all eval rows filtered by unknown labels — skipping.")
                continue

            print(f"\n{_DIVIDER}")
            print(f"  eval_year={eval_year}  bootstrap={len(df_bootstrap):,} rows  test={len(df_eval):,} rows")

            # ── Layer 1 baseline: empty store, very high min_history ──────────
            results_l1 = predict_batch(
                df_eval, layer1_model, embedder, {},
                k=l2["k"],
                threshold=l2["similarity_threshold"],
                min_history=999,
            )
            l1_weighted_f1 = f1_score(
                results_l1["true_label"], results_l1["pred_label"],
                average="weighted", zero_division=0,
            )

            # ── Build Layer 2 store from bootstrap window ─────────────────────
            if len(df_bootstrap) == 0:
                store = {}
            else:
                store = _build_store_in_memory(df_bootstrap, embedder)

            # ── Full pipeline ─────────────────────────────────────────────────
            results = predict_batch(
                df_eval, layer1_model, embedder, store,
                k=l2["k"],
                threshold=l2["similarity_threshold"],
                min_history=l2["min_history"],
            )
            y_true = results["true_label"].tolist()
            y_pred = results["pred_label"].tolist()

            weighted_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            macro_f1      = f1_score(y_true, y_pred, average="macro",    zero_division=0)
            layer2_count  = int((results["source"] == "layer2").sum())
            layer2_pct    = layer2_count / len(results) * 100

            print(f"  Layer 1 only  : weighted_f1={l1_weighted_f1:.4f}")
            print(f"  Layer 1+2     : weighted_f1={weighted_f1:.4f}  (layer2_routing={layer2_pct:.1f}%)")
            print(_DIVIDER)

            # ── Log per-year metrics ──────────────────────────────────────────
            mlflow.log_metric("weighted_f1",             weighted_f1,       step=eval_year)
            mlflow.log_metric("macro_f1",                macro_f1,          step=eval_year)
            mlflow.log_metric("layer2_routing_pct",      layer2_pct,        step=eval_year)
            mlflow.log_metric("bootstrap_size",          len(df_bootstrap), step=eval_year)
            mlflow.log_metric("eval_size",               len(df_eval),      step=eval_year)
            mlflow.log_metric("layer1_only_weighted_f1", l1_weighted_f1,    step=eval_year)

            result_rows.append({
                "eval_year":            eval_year,
                "bootstrap_size":       len(df_bootstrap),
                "eval_size":            len(df_eval),
                "layer1_weighted_f1":   round(l1_weighted_f1, 4),
                "pipeline_weighted_f1": round(weighted_f1,    4),
                "pipeline_macro_f1":    round(macro_f1,       4),
                "layer2_routing_pct":   round(layer2_pct,     2),
            })

        # ── Final summary table ────────────────────────────────────────────────
        if result_rows:
            df_results = pd.DataFrame(result_rows)

            print(f"\n{'═' * 78}")
            print(f"  SLIDING WINDOW SUMMARY")
            print(f"{'═' * 78}")
            print(f"  {'year':>6}  {'bootstrap':>10}  {'eval':>6}  "
                  f"{'l1_f1':>8}  {'pipeline_f1':>12}  {'l2_routing%':>12}")
            print("  " + "─" * 66)
            for _, r in df_results.iterrows():
                print(
                    f"  {int(r['eval_year']):>6}  "
                    f"{int(r['bootstrap_size']):>10,}  "
                    f"{int(r['eval_size']):>6,}  "
                    f"{r['layer1_weighted_f1']:>8.4f}  "
                    f"{r['pipeline_weighted_f1']:>12.4f}  "
                    f"{r['layer2_routing_pct']:>11.1f}%"
                )
            print(f"{'═' * 78}")

            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = os.path.join(tmpdir, "sliding_window_results.csv")
                df_results.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path)
                print(f"\n[mlflow] Artifact logged: sliding_window_results.csv")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
