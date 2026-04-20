"""
test_retrain.py
───────────────
Local test suite for retrain.py.

What is mocked / what is real:
  - MinIO       → mocked (not reachable locally); fget_object copies local data/raw/train.csv
  - MLflow      → real MLflow client, but MLFLOW_TRACKING_URI=file:///tmp/test_mlruns
                  redirects it to a local file store (no server needed)
  - config.yaml → loaded as-is; no path overrides in test code
  - data paths  → resolved via Docker volume mounts at runtime

Build (from project root):
    docker build -f training/Dockerfile.retrain -t categorizer-retrain training/

Run (from project root):
    mkdir -p /tmp/retrain_test_processed

    docker run --rm \\
      -v $(pwd)/training:/app/training \\
      -v $(pwd)/data:/app/data \\
      -v /tmp/retrain_test_processed:/app/data/processed \\
      -e MLFLOW_TRACKING_URI=file:///tmp/test_mlruns \\
      --entrypoint python \\
      categorizer-retrain \\
      test_retrain.py

    # PowerShell:
    New-Item -ItemType Directory -Force /tmp/retrain_test_processed

    docker run --rm `
      -v ${PWD}/training:/app/training `
      -v ${PWD}/data:/app/data `
      -v /tmp/retrain_test_processed:/app/data/processed `
      -e MLFLOW_TRACKING_URI=file:///tmp/test_mlruns `
      --entrypoint python `
      categorizer-retrain `
      test_retrain.py

Volume layout inside the container:
  /app/training          ← training code + config.yaml (read-write, mounted)
  /app/data/raw/         ← source CSVs (read-only in practice)
  /app/data/processed/   ← isolated temp dir so test writes never touch real splits
  /tmp/test_mlruns       ← MLflow file store (lives inside container, discarded on exit)
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import retrain  # noqa: E402

# Inside the container data/raw/train.csv is at /app/data/raw/train.csv.
# os.path.dirname(__file__) == /app/training, so ../data resolves correctly
# whether the test is run inside Docker or directly from training/.
LOCAL_TRAIN_CSV = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/raw/train.csv")
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _FakeObj:
    """Minimal stand-in for a MinIO Object returned by list_objects()."""
    def __init__(self, object_name: str) -> None:
        self.object_name = object_name


def _fake_fget(src_csv: str):
    """Side-effect for client.fget_object — copies a local CSV to dest_path."""
    def fget_object(bucket: str, object_name: str, dest_path: str) -> None:
        shutil.copy(src_csv, dest_path)
    return fget_object


def _mock_client_with_data() -> MagicMock:
    client = MagicMock()
    client.list_objects.return_value = [
        _FakeObj("retraining/retraining_dataset_v20240315_120000.csv"),
    ]
    client.fget_object.side_effect = _fake_fget(LOCAL_TRAIN_CSV)
    return client


def _mock_client_empty() -> MagicMock:
    client = MagicMock()
    client.list_objects.return_value = []
    return client


# ── Unit tests — pure logic ───────────────────────────────────────────────────

class TestDiscoverLatestDataset(unittest.TestCase):
    """discover_latest_dataset is pure logic — no I/O, no minio config needed."""

    _cfg = {"minio": {"bucket": "data"}}

    def test_returns_none_when_bucket_empty(self):
        client = MagicMock()
        client.list_objects.return_value = []
        self.assertIsNone(retrain.discover_latest_dataset(client, self._cfg))

    def test_returns_none_when_no_matching_filenames(self):
        client = MagicMock()
        client.list_objects.return_value = [
            _FakeObj("retraining/README.txt"),
            _FakeObj("retraining/some_other.csv"),
        ]
        self.assertIsNone(retrain.discover_latest_dataset(client, self._cfg))

    def test_picks_lexicographically_last_file(self):
        client = MagicMock()
        client.list_objects.return_value = [
            _FakeObj("retraining/retraining_dataset_v20240101_120000.csv"),
            _FakeObj("retraining/retraining_dataset_v20240315_083000.csv"),
            _FakeObj("retraining/retraining_dataset_v20240201_090000.csv"),
        ]
        result = retrain.discover_latest_dataset(client, self._cfg)
        self.assertEqual(result, "retraining/retraining_dataset_v20240315_083000.csv")

    def test_ignores_non_matching_files_alongside_valid_ones(self):
        client = MagicMock()
        client.list_objects.return_value = [
            _FakeObj("retraining/README.txt"),
            _FakeObj("retraining/retraining_dataset_v20240101_120000.csv"),
            _FakeObj("retraining/other.csv"),
        ]
        result = retrain.discover_latest_dataset(client, self._cfg)
        self.assertEqual(result, "retraining/retraining_dataset_v20240101_120000.csv")

    def test_single_file_returned_directly(self):
        client = MagicMock()
        client.list_objects.return_value = [
            _FakeObj("retraining/retraining_dataset_v20240101_000000.csv"),
        ]
        result = retrain.discover_latest_dataset(client, self._cfg)
        self.assertEqual(result, "retraining/retraining_dataset_v20240101_000000.csv")


# ── Integration tests — full main() with mocked MinIO ─────────────────────────

class TestRetrain(unittest.TestCase):

    def test_exits_zero_when_no_datasets_in_minio(self):
        """
        Empty MinIO bucket → main() prints a helpful message and exits 0.
        Nothing is trained; MLflow has no run written.
        """
        with patch("retrain.make_minio_client", return_value=_mock_client_empty()), \
             patch("sys.argv", ["retrain.py", "--config", "config.yaml"]):

            with self.assertRaises(SystemExit) as cm:
                retrain.main()

            self.assertEqual(cm.exception.code, 0,
                             "Expected exit(0) when no retraining files found")

    @unittest.skipUnless(
        os.path.exists(LOCAL_TRAIN_CSV),
        f"Skipped: {LOCAL_TRAIN_CSV} not mounted — pass -v $(pwd)/data:/app/data",
    )
    def test_full_retrain_pipeline_passes_quality_gate(self):
        """
        Full pipeline using local data/raw/train.csv as the retraining dataset.
        MinIO is mocked; MLflow writes to file:///tmp/test_mlruns via env var.
        tfidf_logreg on 59 k rows clears the quality gate comfortably.
        Verifies: processed splits written, model artifact saved, MLflow run tagged correctly.
        """
        import mlflow

        with patch("retrain.make_minio_client", return_value=_mock_client_with_data()), \
             patch("sys.argv", ["retrain.py", "--config", "config.yaml"]):

            try:
                retrain.main()
            except SystemExit as exc:
                self.fail(
                    f"retrain.main() raised SystemExit({exc.code}).\n"
                    "Quality gate may have failed — check the output above."
                )

        # Processed splits must exist (written by run_preprocessing to data/processed/)
        processed_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/processed")
        )
        for fname in ("train.csv", "val.csv"):
            self.assertTrue(
                os.path.exists(os.path.join(processed_dir, fname)),
                f"Expected {fname} in {processed_dir} after retrain",
            )

        # Model artifact must exist
        artifact = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models/layer1/artifacts/tfidf_logreg.joblib",
        )
        self.assertTrue(os.path.exists(artifact),
                        f"Expected tfidf_logreg.joblib at {artifact}")

        # MLflow run must be recorded with correct tags
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if tracking_uri.startswith("file://"):
            runs = mlflow.search_runs(
                experiment_names=["layer1-retraining"],
                filter_string="tags.run_type = 'retraining'",
            )
            self.assertGreater(len(runs), 0, "Expected at least one MLflow run")
            latest = runs.sort_values("start_time", ascending=False).iloc[0]
            self.assertEqual(latest["tags.quality_gate"], "passed")
            self.assertEqual(
                latest["tags.retraining_dataset"],
                "retraining_dataset_v20240315_120000.csv",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
