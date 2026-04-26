"""
model_pipeline/evaluate.py

Batch evaluation of the full Layer 1 + Layer 2 pipeline.

Default mode: mirrors the 70/30 temporal per-user split from build_store.py —
the store is built from the first 70% of each user's history; evaluation runs
on the last 30%.

With --eval-csv: loads a pre-split evaluation CSV directly from MinIO (e.g.
processed/eval_cex.csv or processed/eval_moneydata.csv), skipping the split.
Rows with categories outside the training label set are dropped and counted.

Metrics logged to MLflow:
  - weighted_f1, macro_f1  (overall)
  - layer2_routing_pct      (% of transactions routed through Layer 2)
  - per-source weighted F1  (layer1_weighted_f1, layer2_weighted_f1)
  - dropped_rows            (only when --eval-csv is used)
  - classification report + per-row results CSV as artifacts

Usage (from project root):
    python -m model_pipeline.evaluate
    python -m model_pipeline.evaluate --config model_pipeline/layer2/config.yaml
    python -m model_pipeline.evaluate --eval-csv processed/eval_moneydata.csv \
        --run-name fasttext-layer2-realworld --experiment-suffix _moneydata
"""

import argparse
import io
import json
import os
import pickle
import tempfile
from urllib.parse import urlparse

import fasttext
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import classification_report, f1_score

from model_pipeline.layer2.build_store import (
    EXTRA_COLS,
    first_n_percent_per_user,
    load_csv,
    make_embed_text,
    make_minio_client,
)
from model_pipeline.layer2.embedder import Embedder
from model_pipeline.layer2.matcher import get_top_k, majority_vote
from training.utils import normalize_payee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model_pipeline/layer2/config.yaml")
    parser.add_argument(
        "--eval-csv", default=None,
        help="MinIO object path of a pre-split eval CSV (e.g. processed/eval_moneydata.csv). "
             "Uses all rows as test data; skips the 70/30 split.",
    )
    parser.add_argument("--run-name", default="layer2-eval")
    parser.add_argument(
        "--experiment-suffix", default="",
        help="Appended to experiment_name from config (e.g. _moneydata).",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_eval_csv(cfg: dict, object_path: str) -> pd.DataFrame:
    client = make_minio_client(cfg)
    response = client.get_object(cfg["minio"]["bucket"], object_path)
    df = pd.read_csv(io.BytesIO(response.read()))
    response.close()
    response.release_conn()
    df.drop(columns=[c for c in EXTRA_COLS if c in df.columns], inplace=True)
    return df


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
    df_test = df_test.copy()
    df_test["_dom"] = pd.to_datetime(df_test["date"]).dt.day
    embed_texts = [
        make_embed_text(row.payee, row.amount, row.day_of_week, row._dom)
        for row in df_test.itertuples(index=False)
    ]

    print(f"  Embedding {len(embed_texts):,} transactions in batch ...")
    embeddings = embedder.embed_batch(embed_texts)

    records = []
    for i, row in enumerate(df_test.itertuples(index=False)):
        labels, probs = layer1_model.predict(normalize_payee(row.payee), k=1)
        l1_cat  = labels[0].replace("__label__", "").replace("_", " ")
        l1_conf = float(probs[0])

        user_data = store.get(row.user_id)
        if user_data is None or len(user_data["labels"]) < min_history:
            records.append({
                "true_label": row.category, "pred_label": l1_cat,
                "confidence": l1_conf, "source": "layer1",
            })
            continue

        neighbors = get_top_k(embeddings[i], user_data, k)
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

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data from MinIO ...")
    if args.eval_csv:
        df_test = _load_eval_csv(cfg, args.eval_csv)
        print(f"Loaded {len(df_test):,} rows from {args.eval_csv}.")
    else:
        df       = load_csv(cfg)
        df_store = first_n_percent_per_user(df, pct=0.70)
        df_test  = df[~df["transaction_id"].isin(df_store["transaction_id"])].copy()
        print(f"Split: {len(df_store):,} store rows  /  {len(df_test):,} test rows")

    # ── Load artifacts ─────────────────────────────────────────────────────────
    print("Loading Layer 1 model from MLflow ...")
    layer1_model = _load_layer1(cfg)

    # ── Drop rows with categories outside the training label set ──────────────
    dropped_rows = 0
    if args.eval_csv:
        known_labels = {
            lbl.replace("__label__", "").replace("_", " ")
            for lbl in layer1_model.get_labels()
        }
        before   = len(df_test)
        df_test  = df_test[df_test["category"].isin(known_labels)].reset_index(drop=True)
        dropped_rows = before - len(df_test)
        if dropped_rows:
            print(f"Dropped {dropped_rows} rows with unknown categories. Remaining: {len(df_test):,}.")

    embedder = Embedder(model_name=l2["model_name"], max_length=l2.get("max_length", 128))

    store_path = l2["store_path"]
    if store_path.startswith("http://") or store_path.startswith("https://"):
        p = urlparse(store_path)
        bucket, obj = p.path.lstrip("/").split("/", 1)
        try:
            response = make_minio_client(cfg).get_object(bucket, obj)
            store = pickle.load(response)
            response.close()
            response.release_conn()
        except Exception as e:
            raise FileNotFoundError(
                f"User store not found at {store_path}. Run build_store.py first."
            ) from e
    else:
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
    experiment_name = cfg["mlflow"]["experiment_name"] + args.experiment_suffix
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].strip())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        git_sha = os.environ.get("GIT_SHA", "")
        if git_sha:
            mlflow.set_tag("git_sha", git_sha)
        if args.eval_csv:
            mlflow.set_tag("dataset", os.path.basename(args.eval_csv))

        mlflow.log_metric("weighted_f1",        weighted_f1)
        mlflow.log_metric("macro_f1",           macro_f1)
        mlflow.log_metric("layer2_routing_pct", round(pct_l2, 2))
        mlflow.log_metric("n_layer1_routed",    n_l1)
        mlflow.log_metric("n_layer2_routed",    n_l2)
        if args.eval_csv:
            mlflow.log_metric("dropped_rows", dropped_rows)
        if "layer1_weighted_f1" in report_dict:
            mlflow.log_metric("layer1_weighted_f1", report_dict["layer1_weighted_f1"])
        if "layer2_weighted_f1" in report_dict:
            mlflow.log_metric("layer2_weighted_f1", report_dict["layer2_weighted_f1"])

        mlflow.log_param("k",                    l2["k"])
        mlflow.log_param("similarity_threshold",  l2["similarity_threshold"])
        mlflow.log_param("min_history",           l2["min_history"])
        mlflow.log_param("layer1_model_uri",      cfg["layer1"]["model_uri"])
        mlflow.log_param("layer1_model_name",     cfg["layer1"].get("model_version", "unknown"))
        if args.eval_csv:
            mlflow.log_param("eval_csv", args.eval_csv)
        layer1_run_id = cfg["layer1"]["model_uri"].split("/")[1] if cfg["layer1"]["model_uri"].startswith("runs:/") else "unknown"
        mlflow.log_param("layer1_run_id",         layer1_run_id)

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
