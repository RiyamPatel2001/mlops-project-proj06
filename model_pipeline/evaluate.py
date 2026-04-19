"""
model_pipeline/evaluate.py

Batch evaluation of the full Layer 1 + Layer 2 pipeline.

Uses the same 70/30 temporal per-user split as build_store.py — the store is
built from the first 70% of each user's history; evaluation runs on the last 30%.

Metrics logged to MLflow:
  - weighted_f1, macro_f1  (overall)
  - layer2_routing_pct      (% of transactions routed through Layer 2)
  - per-source weighted F1  (layer1_weighted_f1, layer2_weighted_f1)
  - classification report + per-row results CSV as artifacts

Usage (from project root):
    python -m model_pipeline.evaluate
    python -m model_pipeline.evaluate --config model_pipeline/layer2/config.yaml
"""

import argparse
import json
import os
import pickle
import tempfile

import fasttext
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import classification_report, f1_score

from model_pipeline.layer2.build_store import first_n_percent_per_user, load_csv
from model_pipeline.layer2.embedder import Embedder
from model_pipeline.layer2.matcher import get_top_k, majority_vote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model_pipeline/layer2/config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_layer1(cfg: dict):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].strip())
    local_path = mlflow.artifacts.download_artifacts(cfg["layer1"]["model_uri"])
    if os.path.isfile(local_path) and local_path.endswith(".bin"):
        return fasttext.load_model(local_path)
    bin_files  = [f for f in os.listdir(local_path) if f.endswith(".bin")]
    if not bin_files:
        raise FileNotFoundError(f"No .bin file found in {local_path}")
    return fasttext.load_model(os.path.join(local_path, bin_files[0]))


def predict_batch(
    df_test: pd.DataFrame,
    layer1_model,
    embedder: Embedder,
    store: dict,
    k: int,
    threshold: float,
    min_history: int,
) -> pd.DataFrame:
    """Run predictions on df_test without mutating the store."""
    records = []
    for row in df_test.itertuples(index=False):
        payee   = row.payee
        user_id = row.user_id

        labels, probs = layer1_model.predict(payee, k=1)
        l1_cat  = labels[0].replace("__label__", "")
        l1_conf = float(probs[0])

        user_data = store.get(user_id)
        if user_data is None or len(user_data["labels"]) < min_history:
            records.append({
                "true_label": row.category, "pred_label": l1_cat,
                "confidence": l1_conf, "source": "layer1",
            })
            continue

        query_emb = embedder.embed(payee)
        neighbors = get_top_k(query_emb, user_data, k)
        l2_cat, l2_conf, exceeded = majority_vote(neighbors, threshold)

        if exceeded:
            records.append({
                "true_label": row.category, "pred_label": l2_cat,
                "confidence": l2_conf, "source": "layer2",
            })
        else:
            records.append({
                "true_label": row.category, "pred_label": l1_cat,
                "confidence": l1_conf, "source": "layer1",
            })

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    l2   = cfg["layer2"]

    # ── Data: mirror the same 70/30 split as build_store ──────────────────────
    print(f"Loading data from MinIO ...")
    df       = load_csv(cfg)
    df_store = first_n_percent_per_user(df, pct=0.70)
    df_test  = df[~df["transaction_id"].isin(df_store["transaction_id"])].copy()
    print(f"Split: {len(df_store):,} store rows  /  {len(df_test):,} test rows")

    # ── Load artifacts ─────────────────────────────────────────────────────────
    print("Loading Layer 1 model from MLflow ...")
    layer1_model = _load_layer1(cfg)

    embedder = Embedder(model_name=l2["model_name"], max_length=l2.get("max_length", 128))

    store_path = l2["store_path"]
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"User store not found at {store_path}. Run build_store.py first."
        )
    with open(store_path, "rb") as f:
        store = pickle.load(f)
    print(f"Loaded store with {len(store)} users.")

    # ── Predict ────────────────────────────────────────────────────────────────
    print(f"Running predictions on {len(df_test):,} test transactions ...")
    results = predict_batch(
        df_test, layer1_model, embedder, store,
        k=l2["k"],
        threshold=l2["similarity_threshold"],
        min_history=l2["min_history"],
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    y_true = results["true_label"].tolist()
    y_pred = results["pred_label"].tolist()

    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    report_str  = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    n_l1   = int((results["source"] == "layer1").sum())
    n_l2   = int((results["source"] == "layer2").sum())
    pct_l2 = n_l2 / len(results) * 100

    for src in ("layer1", "layer2"):
        mask = results["source"] == src
        if mask.sum() == 0:
            continue
        src_f1 = f1_score(
            results.loc[mask, "true_label"],
            results.loc[mask, "pred_label"],
            average="weighted", zero_division=0,
        )
        report_dict[f"{src}_weighted_f1"] = round(float(src_f1), 4)

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Weighted F1   : {weighted_f1:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Routing       : {n_l1} layer1  /  {n_l2} layer2  ({pct_l2:.1f}% L2)")
    print(f"{'─'*55}")
    print(report_str)

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].strip())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="layer2-eval"):
        mlflow.log_metric("weighted_f1",        weighted_f1)
        mlflow.log_metric("macro_f1",           macro_f1)
        mlflow.log_metric("layer2_routing_pct", round(pct_l2, 2))
        mlflow.log_metric("n_layer1_routed",    n_l1)
        mlflow.log_metric("n_layer2_routed",    n_l2)
        if "layer1_weighted_f1" in report_dict:
            mlflow.log_metric("layer1_weighted_f1", report_dict["layer1_weighted_f1"])
        if "layer2_weighted_f1" in report_dict:
            mlflow.log_metric("layer2_weighted_f1", report_dict["layer2_weighted_f1"])

        mlflow.log_param("k",                    l2["k"])
        mlflow.log_param("similarity_threshold",  l2["similarity_threshold"])
        mlflow.log_param("min_history",           l2["min_history"])
        mlflow.log_param("layer1_model_uri",      cfg["layer1"]["model_uri"])

        with tempfile.TemporaryDirectory() as tmp:
            json_path    = os.path.join(tmp, "layer2_eval_report.json")
            txt_path     = os.path.join(tmp, "layer2_eval_report.txt")
            results_path = os.path.join(tmp, "layer2_eval_results.csv")

            with open(json_path, "w") as fh:
                json.dump(report_dict, fh, indent=2)
            with open(txt_path, "w") as fh:
                fh.write(report_str)
            results.to_csv(results_path, index=False)

            mlflow.log_artifact(json_path)
            mlflow.log_artifact(txt_path)
            mlflow.log_artifact(results_path)

        print("[mlflow] Logged metrics and artifacts.")


if __name__ == "__main__":
    main()
