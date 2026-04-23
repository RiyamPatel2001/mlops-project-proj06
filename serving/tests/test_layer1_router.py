from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import app.layer1 as layer1_module
from app.layer1 import (
    ModelRuntime,
    MultiModelRegistry,
    RequestMode,
    Tier,
    TierConfig,
)


def _runtime(
    tier: Tier,
    name: str,
    ready: bool = True,
    blocker: threading.Event | None = None,
    block_on: str = "hold",
) -> ModelRuntime:
    def predict(text: str) -> tuple[str, float]:
        if blocker is not None and text == block_on:
            blocker.wait(timeout=2.0)
        return name, 0.9

    return ModelRuntime(
        config=TierConfig(
            tier=tier,
            name=name,
            kind="test",
            run_id=f"{name}-run",
            artifact_path=f"{name}.bin",
        ),
        predictor=predict,
        version=f"{name}-v1",
        ready=ready,
    )


def _registry(overload_threshold: int = 24) -> MultiModelRegistry:
    registry = MultiModelRegistry(
        configs=[
            TierConfig(Tier.GOOD, "minilm", "test", "good-run", "good"),
            TierConfig(Tier.FAST, "fasttext", "test", "fast-run", "fast"),
            TierConfig(Tier.CHEAP, "tfidf", "test", "cheap-run", "cheap"),
        ],
        request_window_seconds=10,
        batch_sticky_ttl_seconds=60,
        overload_inflight_threshold=overload_threshold,
        overload_sustain_seconds=0.0,
    )
    registry.install_runtime_for_tests(Tier.GOOD, _runtime(Tier.GOOD, "minilm"))
    registry.install_runtime_for_tests(Tier.FAST, _runtime(Tier.FAST, "fasttext"))
    registry.install_runtime_for_tests(Tier.CHEAP, _runtime(Tier.CHEAP, "tfidf"))
    return registry


class Layer1RouterTests(unittest.TestCase):
    def test_interactive_requests_stay_on_good(self) -> None:
        registry = _registry()

        first = registry.predict("merchant", request_mode=RequestMode.INTERACTIVE.value)
        second = registry.predict("merchant", request_mode="unknown-mode")

        self.assertEqual(first.tier, "good")
        self.assertEqual(second.tier, "good")

    def test_bulk_batch_is_pinned_to_fast_by_default(self) -> None:
        registry = _registry()

        first = registry.predict(
            "merchant",
            request_mode=RequestMode.BULK.value,
            batch_id="import-1",
        )
        second = registry.predict(
            "merchant",
            request_mode=RequestMode.BULK.value,
            batch_id="import-1",
        )

        self.assertEqual(first.tier, "fast")
        self.assertEqual(second.tier, "fast")
        self.assertEqual(registry.get_router_snapshot().active_batch_count, 1)

    def test_new_bulk_batches_fall_back_to_cheap_under_overload(self) -> None:
        registry = _registry(overload_threshold=1)
        blocker = threading.Event()
        registry.install_runtime_for_tests(
            Tier.FAST,
            _runtime(Tier.FAST, "fasttext", blocker=blocker),
        )

        results: list[str] = []

        def run_blocked_batch() -> None:
            result = registry.predict(
                "hold",
                request_mode=RequestMode.BULK.value,
                batch_id="import-1",
            )
            results.append(result.tier)

        thread = threading.Thread(target=run_blocked_batch)
        thread.start()

        deadline = time.time() + 2.0
        while time.time() < deadline:
            snapshot = registry.get_router_snapshot()
            if snapshot.total_inflight_requests > 0:
                break
            time.sleep(0.01)
        else:
            self.fail("bulk request never became active")

        overloaded_batch = registry.predict(
            "merchant",
            request_mode=RequestMode.BULK.value,
            batch_id="import-2",
        )
        sticky_batch = registry.predict(
            "merchant",
            request_mode=RequestMode.BULK.value,
            batch_id="import-1",
        )

        blocker.set()
        thread.join(timeout=2.0)

        self.assertEqual(results, ["fast"])
        self.assertEqual(overloaded_batch.tier, "cheap")
        self.assertEqual(sticky_batch.tier, "fast")

    def test_interactive_requests_keep_good_during_bulk_overload(self) -> None:
        registry = _registry(overload_threshold=1)
        blocker = threading.Event()
        registry.install_runtime_for_tests(
            Tier.FAST,
            _runtime(Tier.FAST, "fasttext", blocker=blocker),
        )

        def run_blocked_batch() -> None:
            registry.predict(
                "hold",
                request_mode=RequestMode.BULK.value,
                batch_id="import-1",
            )

        thread = threading.Thread(target=run_blocked_batch)
        thread.start()

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if registry.get_router_snapshot().total_inflight_requests > 0:
                break
            time.sleep(0.01)
        else:
            self.fail("bulk request never became active")

        interactive = registry.predict(
            "merchant",
            request_mode=RequestMode.INTERACTIVE.value,
        )

        blocker.set()
        thread.join(timeout=2.0)

        self.assertEqual(interactive.tier, "good")


class TierConfigResolutionTests(unittest.TestCase):
    @mock.patch("app.layer1._load_registry_overrides")
    @mock.patch("app.layer1._latest_finished_run_id")
    @mock.patch("app.layer1._run_is_servable")
    def test_invalid_registry_run_falls_back_to_latest_same_model(
        self,
        mock_run_is_servable: mock.Mock,
        mock_latest_finished_run_id: mock.Mock,
        mock_load_registry_overrides: mock.Mock,
    ) -> None:
        mock_load_registry_overrides.return_value = {
            "fast": {
                "model_name": "fasttext",
                "model_kind": "fasttext",
                "run_id": "stale-fast-run",
                "artifact_path": "fasttext.bin",
            }
        }
        mock_run_is_servable.side_effect = (
            lambda run_id, model_name: run_id != "stale-fast-run"
        )
        mock_latest_finished_run_id.side_effect = {
            "minilm": "latest-minilm-run",
            "fasttext": "latest-fasttext-run",
            "tfidf_logreg": "latest-tfidf-run",
        }.__getitem__

        configs = layer1_module._build_tier_configs()
        by_tier = {cfg.tier: cfg for cfg in configs}

        self.assertEqual(by_tier[Tier.FAST].run_id, "latest-fasttext-run")
        self.assertEqual(by_tier[Tier.FAST].name, "fasttext")
        self.assertEqual(by_tier[Tier.GOOD].run_id, "latest-minilm-run")
        self.assertEqual(by_tier[Tier.CHEAP].run_id, "latest-tfidf-run")

    @mock.patch("app.layer1._load_registry_overrides")
    @mock.patch("app.layer1._latest_finished_run_id")
    @mock.patch("app.layer1._run_is_servable", return_value=True)
    def test_valid_registry_run_is_preserved(
        self,
        _mock_run_is_servable: mock.Mock,
        mock_latest_finished_run_id: mock.Mock,
        mock_load_registry_overrides: mock.Mock,
    ) -> None:
        mock_load_registry_overrides.return_value = {
            "cheap": {
                "model_name": "tf_idf_logreg",
                "model_kind": "sklearn",
                "run_id": "registry-cheap-run",
                "artifact_path": "tfidf_logreg.joblib",
            }
        }
        mock_latest_finished_run_id.side_effect = {
            "minilm": "latest-minilm-run",
            "fasttext": "latest-fasttext-run",
            "tfidf_logreg": "latest-tfidf-run",
        }.__getitem__

        configs = layer1_module._build_tier_configs()
        by_tier = {cfg.tier: cfg for cfg in configs}

        self.assertEqual(by_tier[Tier.CHEAP].run_id, "registry-cheap-run")
        self.assertEqual(by_tier[Tier.CHEAP].name, "tfidf_logreg")


if __name__ == "__main__":
    unittest.main()
