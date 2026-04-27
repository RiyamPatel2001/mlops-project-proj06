"""
retrain.py
----------
Triggered retraining entrypoint for the Layer-1 categorizer.

Workflow:
  1. Download the latest versioned retraining dataset from MinIO.
  2. Merge it with the base raw dataset (optional) and retrain.
  3. Evaluate the candidate and log accuracy/F1 metrics to MLflow.
  4. Compare the candidate accuracy against the active production model.
  5. Promote the candidate only if it improves on production accuracy.

If no retraining dataset exists yet, the job exits cleanly.
The active model pointer is stored in a shared JSON registry file. The serving
container watches that file and hot-reloads newly promoted MLflow artifacts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
import evaluate as eval_module
import train as train_module
import utils as preprocess_utils
from minio import Minio

_MODEL_ARTIFACT_PATHS = {
    "tfidf_logreg": "tfidf_logreg.joblib",
    "fasttext": "fasttext.bin",
    "minilm": "minilm",
    "distilbert": "distilbert",
    "mpnet": "mpnet",
}

_MODEL_KINDS = {
    "tfidf_logreg": "sklearn",
    "fasttext": "fasttext",
    "minilm": "hf",
    "distilbert": "hf",
    "mpnet": "hf",
}

_MODEL_TARGET_TIERS = {
    "minilm": "good",
    "distilbert": "good",
    "mpnet": "good",
    "fasttext": "fast",
    "tfidf_logreg": "cheap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain Layer-1 categorizer with new data from MinIO."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging with the base raw dataset; use retraining data only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the latest dataset version is already active.",
    )
    return parser.parse_args()


def make_minio_client(cfg: dict) -> Minio:
    minio_cfg = cfg["minio"]
    endpoint_url = os.environ.get("MINIO_ENDPOINT_URL", minio_cfg["endpoint"])
    endpoint = endpoint_url.replace("http://", "").replace("https://", "")
    secure = endpoint_url.startswith("https://")
    return Minio(
        endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin123"),
        secure=secure,
    )


def _json_loads(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode("utf-8"))


def _read_object_bytes(client: Minio, bucket: str, object_name: str) -> bytes:
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _extract_version_from_name(name: str) -> str | None:
    match = re.search(r"_v([0-9]{8}_[0-9]{6})\.(?:csv|json)$", name)
    return match.group(1) if match else None


def _select_latest_manifest(
    client: Minio,
    bucket: str,
    prefix: str,
) -> tuple[str, dict[str, Any]] | None:
    manifests: list[tuple[str, str]] = []
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith(".json"):
            continue
        version = _extract_version_from_name(obj.object_name)
        if version:
            manifests.append((version, obj.object_name))

    if not manifests:
        return None

    version, object_name = max(manifests, key=lambda item: item[0])
    manifest = _json_loads(_read_object_bytes(client, bucket, object_name))
    manifest.setdefault("version", version)
    manifest.setdefault("manifest_path", f"s3://{bucket}/{object_name}")
    return object_name, manifest


def _load_dataset_from_manifest(
    client: Minio,
    bucket: str,
    prefix: str,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_uri = str(manifest.get("minio_path", "")).strip()
    if dataset_uri.startswith(f"s3://{bucket}/"):
        dataset_key = dataset_uri[len(f"s3://{bucket}/") :]
    else:
        version = manifest["version"]
        dataset_key = f"{prefix}retraining_dataset_v{version}.csv"
        dataset_uri = f"s3://{bucket}/{dataset_key}"

    csv_bytes = _read_object_bytes(client, bucket, dataset_key)
    df = pd.read_csv(io.BytesIO(csv_bytes))

    metadata = {
        "version": manifest["version"],
        "dataset_key": dataset_key,
        "dataset_uri": dataset_uri,
        "manifest_path": manifest.get("manifest_path"),
        "record_count": int(manifest.get("total_records", len(df))),
    }
    return df, metadata


def _normalize_retraining_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    rename_map = {}
    if "label" in frame.columns and "category" not in frame.columns:
        rename_map["label"] = "category"
    if rename_map:
        frame = frame.rename(columns=rename_map)

    if "payee" not in frame.columns and "feature_vector" in frame.columns:
        frame["payee"] = (
            frame["feature_vector"]
            .astype(str)
            .str.split(" | ", n=1, regex=False)
            .str[0]
            .str.strip()
        )

    required = {"payee", "category", "user_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Retraining dataset is missing required columns: " + ", ".join(missing)
        )

    if "transaction_id" not in frame.columns:
        frame["transaction_id"] = ""

    keep = ["transaction_id", "user_id", "payee", "category"]
    return frame[keep].dropna(subset=["payee", "category", "user_id"])


def download_latest_retraining_data(
    client: Minio,
    cfg: dict,
) -> tuple[pd.DataFrame, dict[str, Any]] | tuple[None, None]:
    bucket = cfg["minio"]["bucket"]
    prefix = cfg["data"]["retraining_prefix"].rstrip("/") + "/"
    selection = _select_latest_manifest(client, bucket, prefix)
    if selection is None:
        return None, None

    _, manifest = selection
    df, metadata = _load_dataset_from_manifest(client, bucket, prefix, manifest)
    normalized = _normalize_retraining_dataframe(df)
    print(
        f"[retrain] Downloaded dataset v{metadata['version']} from {metadata['dataset_uri']} "
        f"({len(normalized):,} rows)"
    )
    return normalized, metadata


def _artifact_path_for_model(model_name: str) -> str:
    if model_name not in _MODEL_ARTIFACT_PATHS:
        raise ValueError(f"Unsupported model artifact mapping for {model_name}")
    return _MODEL_ARTIFACT_PATHS[model_name]


def _model_kind_for_name(model_name: str) -> str:
    if model_name not in _MODEL_KINDS:
        raise ValueError(f"Unsupported model kind mapping for {model_name}")
    return _MODEL_KINDS[model_name]


def _promotion_cfg(cfg: dict) -> dict[str, Any]:
    return cfg.get("promotion", {})


def _target_tier(cfg: dict) -> str:
    configured = str(_promotion_cfg(cfg).get("target_tier", "")).strip().lower()
    if configured:
        return configured

    model_name = str(cfg.get("model", "")).strip().lower()
    if model_name in _MODEL_TARGET_TIERS:
        return _MODEL_TARGET_TIERS[model_name]

    raise RuntimeError(f"No default serving tier is defined for model {model_name}")


def _registry_path(cfg: dict) -> str:
    promotion = _promotion_cfg(cfg)
    return os.environ.get(
        "MODEL_REGISTRY_PATH",
        str(promotion.get("registry_path", "../serving/layer1_registry.json")),
    )


def _minimum_accuracy_delta(cfg: dict) -> float:
    return float(_promotion_cfg(cfg).get("minimum_accuracy_delta", 0.0))


def _comparison_eval_csv(cfg: dict) -> str | None:
    value = str(_promotion_cfg(cfg).get("comparison_eval_csv", "")).strip()
    return value or None


def _load_registry(path: str) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {"tiers": {}}

    with registry_path.open() as handle:
        content = json.load(handle)
    if "tiers" not in content or not isinstance(content["tiers"], dict):
        raise RuntimeError(f"Invalid model registry file at {path}")
    return content


def _write_registry(path: str, registry: dict[str, Any]) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_path.with_suffix(".tmp")
    with tmp_path.open("w") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
    tmp_path.replace(registry_path)


def _bootstrap_reference(cfg: dict) -> dict[str, Any] | None:
    bootstrap = _promotion_cfg(cfg).get("bootstrap", {})
    if not bootstrap:
        return None
    required = {"run_id", "artifact_path", "model_name", "model_kind"}
    if not required.issubset(bootstrap):
        raise RuntimeError(
            "promotion.bootstrap must define run_id, artifact_path, model_name, and model_kind"
        )
    return dict(bootstrap)


def _active_reference(cfg: dict, registry: dict[str, Any]) -> dict[str, Any] | None:
    tier = _target_tier(cfg)
    tiers = registry.get("tiers", {})
    if tier in tiers:
        return dict(tiers[tier])
    return _bootstrap_reference(cfg)


def _resolve_fasttext_path(local_path: str) -> str:
    path = Path(local_path)
    if path.is_file():
        return str(path)
    for suffix in (".bin", ".ftz"):
        matches = sorted(path.rglob(f"*{suffix}"))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"Could not find a FastText model beneath {local_path}")


def _label_classes_from_mlflow(run_id: str) -> list[str] | None:
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="label_classes.json",
        )
    except Exception:
        return None

    try:
        with open(local_path) as handle:
            payload = json.load(handle)
    except Exception:
        return None

    return payload if isinstance(payload, list) else None


def _normalize_comparison_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "label" in frame.columns and "category" not in frame.columns:
        frame = frame.rename(columns={"label": "category"})

    required = {"payee", "category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Comparison dataset is missing required columns: " + ", ".join(missing)
        )

    frame = frame.dropna(subset=["payee", "category"]).copy()
    frame["payee_norm"] = frame["payee"].astype(str).map(preprocess_utils.normalize_payee)
    return frame


def _load_comparison_dataset(
    client: Minio,
    cfg: dict,
    label_classes: list[str],
) -> tuple[pd.Series, list[int], dict[str, Any]] | None:
    object_name = _comparison_eval_csv(cfg)
    if object_name is None:
        return None

    csv_bytes = _read_object_bytes(client, cfg["minio"]["bucket"], object_name)
    frame = _normalize_comparison_dataframe(pd.read_csv(io.BytesIO(csv_bytes)))
    label_to_index = {label: index for index, label in enumerate(label_classes)}
    frame = frame[frame["category"].isin(label_to_index)].copy()
    if frame.empty:
        raise RuntimeError(
            "Comparison dataset has no rows whose categories match the current label classes"
        )

    y_compare = [label_to_index[label] for label in frame["category"].tolist()]
    metadata = {
        "comparison_eval_csv": object_name,
        "comparison_row_count": len(frame),
    }
    return frame["payee_norm"], y_compare, metadata


def _candidate_accuracy(vec, clf, X_eval, y_eval) -> float:
    if vec is not None:
        predictions = clf.predict(vec.transform(X_eval))
    else:
        predictions = clf.predict(X_eval)
    return float(accuracy_score(y_eval, predictions))


def _accuracy_for_fasttext(
    ref: dict[str, Any],
    X_val,
    y_val,
    label_classes: list[str],
) -> float:
    import fasttext

    local_path = mlflow.artifacts.download_artifacts(
        run_id=ref["run_id"],
        artifact_path=ref["artifact_path"],
    )
    model_path = _resolve_fasttext_path(local_path)
    ft_model = fasttext.load_model(model_path)
    label_to_index = {
        f"__label__{label.replace(' ', '_')}": index
        for index, label in enumerate(label_classes)
    }

    predictions = []
    for text in X_val.tolist() if hasattr(X_val, "tolist") else X_val:
        labels, _ = ft_model.predict(str(text).replace("\n", " "), k=1)
        predictions.append(label_to_index.get(labels[0], 0))

    return float(accuracy_score(y_val, predictions))


def _accuracy_for_sklearn(ref: dict[str, Any], X_val, y_val) -> float:
    import joblib

    local_path = mlflow.artifacts.download_artifacts(
        run_id=ref["run_id"],
        artifact_path=ref["artifact_path"],
    )
    payload = joblib.load(local_path)
    vectorizer = None
    classifier = payload
    if isinstance(payload, dict):
        vectorizer = payload.get("vectorizer")
        classifier = payload.get("classifier", payload)

    if vectorizer is not None:
        features = vectorizer.transform(X_val)
        predictions = classifier.predict(features)
    else:
        predictions = classifier.predict(X_val)
    return float(accuracy_score(y_val, predictions))


def _accuracy_for_hf(
    ref: dict[str, Any],
    X_val,
    y_val,
    label_classes: list[str],
) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_dir = mlflow.artifacts.download_artifacts(
        run_id=ref["run_id"],
        artifact_path=ref["artifact_path"],
    )
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(local_dir)
    model.eval()

    ref_label_classes = _label_classes_from_mlflow(ref["run_id"]) or list(label_classes)
    current_label_lookup = {
        label: index for index, label in enumerate(label_classes)
    }

    predictions = []
    for text in X_val.tolist() if hasattr(X_val, "tolist") else X_val:
        inputs = tokenizer(
            str(text),
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_index = int(torch.argmax(logits, dim=-1).item())
        predicted_label = (
            ref_label_classes[predicted_index]
            if predicted_index < len(ref_label_classes)
            else "Other"
        )
        predictions.append(current_label_lookup.get(predicted_label, 0))

    return float(accuracy_score(y_val, predictions))


def _production_accuracy(
    ref: dict[str, Any] | None,
    X_val,
    y_val,
    label_classes: list[str],
) -> float | None:
    if not ref:
        return None

    kind = str(ref.get("model_kind", "")).strip().lower()
    if kind == "fasttext":
        return _accuracy_for_fasttext(ref, X_val, y_val, label_classes)
    if kind == "sklearn":
        return _accuracy_for_sklearn(ref, X_val, y_val)
    if kind == "hf":
        return _accuracy_for_hf(ref, X_val, y_val, label_classes)

    raise RuntimeError(f"Unsupported production model kind for comparison: {kind}")


def _should_promote(
    candidate_accuracy: float,
    production_accuracy: float | None,
    cfg: dict,
) -> bool:
    if production_accuracy is None:
        return True
    return candidate_accuracy >= production_accuracy + _minimum_accuracy_delta(cfg)


def _candidate_reference(
    cfg: dict,
    run_id: str,
    metrics: dict[str, float],
    dataset_meta: dict[str, Any],
) -> dict[str, Any]:
    model_name = cfg["model"]
    artifact_path = _artifact_path_for_model(model_name)
    return {
        "model_name": model_name,
        "model_kind": _model_kind_for_name(model_name),
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri": f"runs:/{run_id}/{artifact_path}",
        "accuracy": round(float(metrics["accuracy"]), 6),
        "weighted_f1": round(float(metrics["weighted_f1"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "dataset_version": dataset_meta["version"],
        "dataset_uri": dataset_meta["dataset_uri"],
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }


def _update_registry(
    cfg: dict,
    registry: dict[str, Any],
    candidate_ref: dict[str, Any],
    registry_path: str,
) -> None:
    tier = _target_tier(cfg)
    registry.setdefault("tiers", {})
    registry["tiers"][tier] = candidate_ref
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    registry["updated_by"] = "training/retrain.py"
    _write_registry(registry_path, registry)


def _latest_active_dataset_version(active_ref: dict[str, Any] | None) -> str | None:
    if not active_ref:
        return None
    version = str(active_ref.get("dataset_version", "")).strip()
    return version or None


def _quality_gate_status() -> str:
    active_run = mlflow.active_run()
    tags = getattr(getattr(active_run, "data", None), "tags", None)
    if isinstance(tags, dict):
        return str(tags.get("quality_gate", "unknown"))

    mlflow_tags = getattr(mlflow, "tags", None)
    if isinstance(mlflow_tags, dict):
        return str(mlflow_tags.get("quality_gate", "unknown"))

    return "unknown"


def main() -> None:
    args = parse_args()
    cfg = train_module.load_config(args.config)

    client = make_minio_client(cfg)
    retrain_df, dataset_meta = download_latest_retraining_data(client, cfg)
    if retrain_df is None or dataset_meta is None:
        print(
            "[retrain] No retraining datasets found in MinIO under "
            f"{cfg['data']['retraining_prefix']}. Nothing to do."
        )
        return

    registry_path = _registry_path(cfg)
    registry = _load_registry(registry_path)
    active_ref = _active_reference(cfg, registry)

    active_dataset_version = _latest_active_dataset_version(active_ref)
    if (
        not args.force
        and active_dataset_version
        and active_dataset_version == dataset_meta["version"]
    ):
        print(
            f"[retrain] Dataset v{dataset_meta['version']} is already the active production dataset. "
            "Use --force to retrain again."
        )
        return

    if not args.no_merge:
        base_path = cfg["data"]["raw_path"]
        try:
            if base_path.startswith("http://") or base_path.startswith("https://"):
                p = urlparse(base_path)
                bucket, obj = p.path.lstrip("/").split("/", 1)
                df_base = pd.read_csv(
                    io.BytesIO(_read_object_bytes(client, bucket, obj))
                )
            else:
                df_base = pd.read_csv(base_path)
            df_combined = pd.concat([df_base, retrain_df], ignore_index=True)
            if "transaction_id" in df_combined.columns:
                df_combined = df_combined.drop_duplicates(subset=["transaction_id"])
            print(
                f"[retrain] Merged base ({len(df_base):,}) + new ({len(retrain_df):,}) "
                f"-> {len(df_combined):,} rows"
            )
        except Exception as exc:
            print(f"[retrain] Could not load base dataset ({exc}); using retraining data only.")
            df_combined = retrain_df
    else:
        df_combined = retrain_df

    with tempfile.TemporaryDirectory() as tmp:
        merged_path = os.path.join(tmp, "retrain_raw.csv")
        df_combined.to_csv(merged_path, index=False)
        cfg["data"]["raw_path"] = merged_path

        comparison_target = "validation_split"
        comparison_rows = 0
        production_accuracy = None
        candidate_accuracy = None
        accuracy_delta = None
        promotion_status = "unknown"

        train_module.setup_mlflow(cfg)

        with mlflow.start_run(run_name=f"{cfg['model']}-retrain"):
            run_id = mlflow.active_run().info.run_id
            mlflow.set_tag("git_sha", train_module.get_git_sha())
            mlflow.set_tag("trigger", "retrain")
            mlflow.set_tag("target_tier", _target_tier(cfg))
            mlflow.set_tag("dataset_version", dataset_meta["version"])
            mlflow.log_param("dataset_uri", dataset_meta["dataset_uri"])
            mlflow.log_param("dataset_manifest_path", dataset_meta["manifest_path"])
            train_module.log_config_params(cfg)

            X_train, X_val, y_train, y_val, label_classes = train_module.run_preprocessing(cfg)

            t0 = time.perf_counter()
            result = train_module.run_training(X_train, y_train, cfg)
            mlflow.log_metric(
                "training_time_seconds",
                round(time.perf_counter() - t0, 2),
            )

            vec, clf = result if isinstance(result, tuple) and len(result) == 2 else (None, result)

            metrics = eval_module.evaluate_and_log(
                clf=clf,
                vec=vec,
                X_val=X_val,
                y_val=y_val,
                label_classes=label_classes,
                config=cfg,
            )

            train_module.save_and_log_model(vec, clf, cfg)

            comparison = _load_comparison_dataset(client, cfg, label_classes)
            if comparison is None:
                compare_X, compare_y = X_val, y_val
                comparison_rows = len(y_val)
                candidate_accuracy = float(metrics["accuracy"])
            else:
                compare_X, compare_y, comparison_meta = comparison
                comparison_target = comparison_meta["comparison_eval_csv"]
                comparison_rows = comparison_meta["comparison_row_count"]
                candidate_accuracy = _candidate_accuracy(vec, clf, compare_X, compare_y)

            production_accuracy = _production_accuracy(
                active_ref,
                compare_X,
                compare_y,
                label_classes,
            )
            accuracy_delta = (
                candidate_accuracy
                if production_accuracy is None
                else candidate_accuracy - production_accuracy
            )

            mlflow.log_metric("candidate_accuracy", candidate_accuracy)
            if production_accuracy is not None:
                mlflow.log_metric("production_accuracy", production_accuracy)
            mlflow.log_metric("accuracy_delta", accuracy_delta)
            mlflow.set_tag("comparison_target", comparison_target)
            mlflow.log_param("comparison_target", comparison_target)
            mlflow.log_param("comparison_row_count", comparison_rows)

            if active_ref is not None:
                mlflow.set_tag("serving_run_id_before", active_ref["run_id"])

            candidate_ref = _candidate_reference(cfg, run_id, metrics, dataset_meta)
            candidate_ref["comparison_target"] = comparison_target
            candidate_ref["comparison_accuracy"] = round(candidate_accuracy, 6)

            if _should_promote(candidate_accuracy, production_accuracy, cfg):
                _update_registry(cfg, registry, candidate_ref, registry_path)
                promotion_status = "promoted"
            else:
                promotion_status = "rejected"

            mlflow.set_tag("promotion_status", promotion_status)

            quality_gate = _quality_gate_status()

        # ── Step 6: Summary ───────────────────────────────────────────────────
        print(
            f"\n{'═'*55}\n"
            f"  Dataset        : {dataset_meta['version']}\n"
            f"  Model          : {cfg['model']}\n"
            f"  Target tier    : {_target_tier(cfg)}\n"
            f"  Compare on     : {comparison_target} ({comparison_rows:,} rows)\n"
            f"  Candidate Acc  : {candidate_accuracy:.4f}\n"
            f"  Production Acc : "
            f"{'n/a' if production_accuracy is None else f'{production_accuracy:.4f}'}\n"
            f"  Accuracy delta : "
            f"{'n/a' if accuracy_delta is None else f'{accuracy_delta:.4f}'}\n"
            f"  Weighted F1    : {metrics['weighted_f1']:.4f}\n"
            f"  Macro F1       : {metrics['macro_f1']:.4f}\n"
            f"  Quality gate   : {quality_gate}\n"
            f"  Promotion      : {promotion_status}\n"
            f"{'═'*55}"
        )


if __name__ == "__main__":
    main()
