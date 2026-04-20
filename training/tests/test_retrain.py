import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class _FakeRunInfo:
    def __init__(self, run_id: str):
        self.run_id = run_id


class _FakeRun:
    def __init__(self, run_id: str):
        self.info = _FakeRunInfo(run_id)


class _FakeRunContext:
    def __init__(self, mlflow_module, run_id: str):
        self._mlflow = mlflow_module
        self._run_id = run_id

    def __enter__(self):
        self._mlflow._active_run = _FakeRun(self._run_id)
        return self._mlflow._active_run

    def __exit__(self, exc_type, exc, tb):
        self._mlflow._active_run = None
        return False


class _FakeMlflow(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self._active_run = None
        self.logged_metrics = {}
        self.logged_params = {}
        self.tags = {}
        self.artifacts = types.SimpleNamespace(download_artifacts=lambda **kwargs: "")

    def start_run(self, run_name=None):
        return _FakeRunContext(self, "candidate-run")

    def active_run(self):
        return self._active_run

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_param(self, key, value):
        self.logged_params[key] = value

    def log_metric(self, key, value):
        self.logged_metrics[key] = value

    def set_tracking_uri(self, uri):
        self.logged_params["tracking_uri"] = uri

    def get_tracking_uri(self):
        return self.logged_params.get("tracking_uri", "http://mlflow.local")

    def set_experiment(self, name):
        self.logged_params["experiment_name"] = name


class _FakeSeries(list):
    def tolist(self):
        return list(self)


class _FakeFrame:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def to_csv(self, path, index=False):
        if not self.rows:
            Path(path).write_text("")
            return
        headers = list(self.rows[0].keys())
        lines = [",".join(headers)]
        for row in self.rows:
            lines.append(",".join(str(row.get(header, "")) for header in headers))
        Path(path).write_text("\n".join(lines) + "\n")


def _load_retrain_module(fake_mlflow, fake_train, fake_evaluate):
    fake_minio = types.SimpleNamespace(Minio=object)
    fake_pandas = types.SimpleNamespace(DataFrame=object, read_csv=lambda *args, **kwargs: None)
    fake_metrics = types.SimpleNamespace(accuracy_score=lambda y_true, y_pred: 1.0)

    patchers = [
        mock.patch.dict(
            sys.modules,
            {
                "mlflow": fake_mlflow,
                "train": fake_train,
                "evaluate": fake_evaluate,
                "minio": fake_minio,
                "pandas": fake_pandas,
                "sklearn": types.SimpleNamespace(metrics=fake_metrics),
                "sklearn.metrics": fake_metrics,
            },
        )
    ]
    for patcher in patchers:
        patcher.start()

    try:
        module_path = Path(__file__).resolve().parents[1] / "retrain.py"
        spec = importlib.util.spec_from_file_location("retrain_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module, patchers
    except Exception:
        for patcher in reversed(patchers):
            patcher.stop()
        raise


class RetrainEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.registry_path = self.root / "layer1_registry.json"
        self.base_csv = self.root / "base.csv"
        self.base_csv.write_text(
            "transaction_id,user_id,payee,category\n"
            "base-1,u1,STORE,Groceries\n"
        )

    def _config(self):
        return {
            "model": "fasttext",
            "data": {
                "raw_path": str(self.base_csv),
                "processed_dir": str(self.root / "processed"),
                "retraining_prefix": "data/retraining/",
            },
            "mlflow": {
                "tracking_uri": "http://mlflow.local",
                "experiment_name": "layer1-categorizer",
            },
            "minio": {
                "endpoint": "http://minio.local:9000",
                "bucket": "data",
            },
            "promotion": {
                "target_tier": "fast",
                "registry_path": str(self.registry_path),
                "minimum_accuracy_delta": 0.0,
                "bootstrap": {
                    "model_name": "fasttext",
                    "model_kind": "fasttext",
                    "run_id": "production-run",
                    "artifact_path": "fasttext.bin",
                },
            },
            "fasttext": {"lr": 0.1, "epoch": 5, "wordNgrams": 2, "dim": 10},
            "val_frac": 0.2,
            "random_state": 42,
        }

    def _dataset(self):
        df = _FakeFrame(
            [
                {"transaction_id": "txn-1", "user_id": "u2", "payee": "COFFEE SHOP", "category": "Dining Out"},
                {"transaction_id": "txn-2", "user_id": "u3", "payee": "MARKET", "category": "Groceries"},
            ]
        )
        meta = {
            "version": "20260420_120000",
            "dataset_uri": "s3://data/data/retraining/retraining_dataset_v20260420_120000.csv",
            "manifest_path": "s3://data/data/retraining/retraining_manifest_v20260420_120000.json",
        }
        return df, meta

    def _train_module(self):
        module = types.SimpleNamespace()
        module.load_config = lambda path: self._config()
        module.setup_mlflow = lambda cfg: None
        module.log_config_params = lambda cfg: None
        module.get_git_sha = lambda: "deadbeef"
        module.run_preprocessing = lambda cfg: (
            _FakeSeries(["COFFEE SHOP", "MARKET"]),
            _FakeSeries(["COFFEE SHOP", "MARKET"]),
            [0, 1],
            [0, 1],
            ["Dining Out", "Groceries"],
        )
        module.run_training = lambda X_train, y_train, cfg: (None, object())
        module.save_and_log_model = lambda vec, clf, cfg: None
        return module

    def _evaluate_module(self, accuracy_value: float):
        module = types.SimpleNamespace()
        module.evaluate_and_log = lambda **kwargs: {
            "accuracy": accuracy_value,
            "weighted_f1": accuracy_value,
            "macro_f1": accuracy_value,
        }
        return module

    def test_promotes_candidate_when_accuracy_improves(self):
        fake_mlflow = _FakeMlflow()
        module, patchers = _load_retrain_module(
            fake_mlflow,
            self._train_module(),
            self._evaluate_module(0.91),
        )
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patchers)])

        dataset_df, dataset_meta = self._dataset()
        module.make_minio_client = lambda cfg: object()
        module.download_latest_retraining_data = lambda client, cfg: (dataset_df, dataset_meta)
        module._production_accuracy = lambda ref, X_val, y_val, label_classes: 0.72

        with mock.patch.object(sys, "argv", ["retrain.py", "--config", "config.yaml", "--no-merge"]):
            module.main()

        self.assertTrue(self.registry_path.exists())
        registry = json.loads(self.registry_path.read_text())
        fast_tier = registry["tiers"]["fast"]
        self.assertEqual(fast_tier["run_id"], "candidate-run")
        self.assertEqual(fast_tier["model_uri"], "runs:/candidate-run/fasttext.bin")
        self.assertEqual(fake_mlflow.tags["promotion_status"], "promoted")
        self.assertAlmostEqual(fake_mlflow.logged_metrics["production_accuracy"], 0.72)
        self.assertAlmostEqual(fake_mlflow.logged_metrics["accuracy_delta"], 0.19, places=6)

    def test_rejects_candidate_when_accuracy_regresses(self):
        self.registry_path.write_text(
            json.dumps(
                {
                    "tiers": {
                        "fast": {
                            "model_name": "fasttext",
                            "model_kind": "fasttext",
                            "run_id": "current-run",
                            "artifact_path": "fasttext.bin",
                            "model_uri": "runs:/current-run/fasttext.bin",
                            "dataset_version": "20260413_120000",
                        }
                    }
                }
            )
        )

        fake_mlflow = _FakeMlflow()
        module, patchers = _load_retrain_module(
            fake_mlflow,
            self._train_module(),
            self._evaluate_module(0.81),
        )
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patchers)])

        dataset_df, dataset_meta = self._dataset()
        module.make_minio_client = lambda cfg: object()
        module.download_latest_retraining_data = lambda client, cfg: (dataset_df, dataset_meta)
        module._production_accuracy = lambda ref, X_val, y_val, label_classes: 0.88

        with mock.patch.object(sys, "argv", ["retrain.py", "--config", "config.yaml", "--no-merge"]):
            module.main()

        registry = json.loads(self.registry_path.read_text())
        fast_tier = registry["tiers"]["fast"]
        self.assertEqual(fast_tier["run_id"], "current-run")
        self.assertEqual(fake_mlflow.tags["promotion_status"], "rejected")


if __name__ == "__main__":
    unittest.main()
