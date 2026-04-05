"""
Download trained models from MLflow server.

Connects to the MLflow tracking server, identifies the 4 completed training
runs (distilbert, minilm, fasttext, tfidf_logreg), and downloads their
artifacts into a local models/ directory.

Usage:
    python download_models.py --tracking-uri http://129.114.26.151:8000 --output-dir ./models
"""

import argparse
import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient


EXPECTED_MODELS = {"distilbert", "minilm", "fasttext", "tfidf_logreg"}


def find_finished_runs(client, experiment_id="1"):
    """Return only FINISHED runs from the given experiment."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    return runs


def identify_model_runs(runs):
    """Match runs to expected model names using run name or tags."""
    matched = {}
    for run in runs:
        run_name = (run.info.run_name or "").lower()
        tags = {k.lower(): v.lower() for k, v in run.data.tags.items()}

        for model_name in EXPECTED_MODELS:
            if model_name in matched:
                continue
            if (
                model_name in run_name
                or model_name in tags.get("mlflow.runname", "")
                or model_name in tags.get("model_type", "")
                or model_name in tags.get("model_name", "")
            ):
                matched[model_name] = run
                break

    return matched


def download_artifacts(client, matched_runs, output_dir):
    """Download model artifacts for each matched run."""
    os.makedirs(output_dir, exist_ok=True)

    for model_name, run in matched_runs.items():
        dest = os.path.join(output_dir, model_name)
        if os.path.exists(dest):
            shutil.rmtree(dest)

        print(f"Downloading {model_name} (run_id={run.info.run_id})...")
        artifact_path = client.download_artifacts(
            run.info.run_id, "", dst_path=dest
        )
        print(f"  -> saved to {artifact_path}")

        params = run.data.params
        metrics = run.data.metrics
        with open(os.path.join(dest, "run_info.txt"), "w") as f:
            f.write(f"run_id: {run.info.run_id}\n")
            f.write(f"run_name: {run.info.run_name}\n")
            f.write(f"status: {run.info.status}\n")
            f.write(f"params: {params}\n")
            f.write(f"metrics: {metrics}\n")

    return matched_runs


def main():
    parser = argparse.ArgumentParser(description="Download models from MLflow")
    parser.add_argument(
        "--tracking-uri",
        default="http://129.114.26.151:8000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to save downloaded models",
    )
    parser.add_argument(
        "--experiment-id",
        default="1",
        help="MLflow experiment ID",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    print(f"Connecting to MLflow at {args.tracking_uri}")
    runs = find_finished_runs(client, args.experiment_id)
    print(f"Found {len(runs)} finished runs")

    matched = identify_model_runs(runs)
    print(f"Matched models: {list(matched.keys())}")

    missing = EXPECTED_MODELS - set(matched.keys())
    if missing:
        print(f"WARNING: Could not find runs for: {missing}")
        print("Available run names:")
        for run in runs:
            print(f"  - {run.info.run_name} (id={run.info.run_id})")

    if matched:
        download_artifacts(client, matched, args.output_dir)
        print("\nDownload complete. Models saved to:")
        for name in sorted(matched.keys()):
            print(f"  {args.output_dir}/{name}/")
    else:
        print("No models matched. Check run names on the MLflow UI.")


if __name__ == "__main__":
    main()
