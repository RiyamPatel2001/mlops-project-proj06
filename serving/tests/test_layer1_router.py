from __future__ import annotations

import threading
import time
import unittest

from app.layer1 import (
    ModelRuntime,
    MultiModelRegistry,
    Tier,
    TierConfig,
)


def _runtime(tier: Tier, name: str, ready: bool = True, blocker: threading.Event | None = None) -> ModelRuntime:
    def predict(_: str) -> tuple[str, float]:
        if blocker is not None:
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


def _registry() -> MultiModelRegistry:
    registry = MultiModelRegistry(
        configs=[
            TierConfig(Tier.GOOD, "minilm", "test", "good-run", "good"),
            TierConfig(Tier.FAST, "fasttext", "test", "fast-run", "fast"),
            TierConfig(Tier.CHEAP, "tfidf", "test", "cheap-run", "cheap"),
        ],
        demand_window_seconds=4,
        medium_demand_rps=0.5,
        high_demand_rps=1.0,
        switch_cooldown_seconds=0.0,
    )
    registry.install_runtime_for_tests(Tier.GOOD, _runtime(Tier.GOOD, "minilm"))
    registry.install_runtime_for_tests(Tier.FAST, _runtime(Tier.FAST, "fasttext"))
    registry.install_runtime_for_tests(Tier.CHEAP, _runtime(Tier.CHEAP, "tfidf"))
    return registry


class Layer1RouterTests(unittest.TestCase):
    def test_low_medium_high_demand_routes_good_fast_cheap(self) -> None:
        registry = _registry()

        first = registry.predict("merchant")
        second = registry.predict("merchant")
        third = registry.predict("merchant")
        fourth = registry.predict("merchant")

        self.assertEqual(first.tier, "good")
        self.assertEqual(second.tier, "fast")
        self.assertEqual(third.tier, "fast")
        self.assertEqual(fourth.tier, "cheap")

    def test_switch_waits_for_ready_target(self) -> None:
        registry = _registry()
        registry.install_runtime_for_tests(
            Tier.FAST,
            _runtime(Tier.FAST, "fasttext", ready=False),
        )

        first = registry.predict("merchant")
        second = registry.predict("merchant")
        snapshot = registry.get_router_snapshot()

        self.assertEqual(first.tier, "good")
        self.assertEqual(second.tier, "good")
        self.assertEqual(snapshot.pending_tier, "fast")
        self.assertEqual(snapshot.active_tier, "good")

    def test_inflight_request_completes_on_old_tier_during_switch(self) -> None:
        registry = _registry()
        blocker = threading.Event()
        registry.install_runtime_for_tests(
            Tier.GOOD,
            _runtime(Tier.GOOD, "minilm", blocker=blocker),
        )

        results: list[str] = []

        def run_blocked_request() -> None:
            results.append(registry.predict("merchant").tier)

        thread = threading.Thread(target=run_blocked_request)
        thread.start()

        deadline = time.time() + 2.0
        while time.time() < deadline:
            snapshot = registry.get_router_snapshot()
            good_status = next(model for model in snapshot.models if model.tier == "good")
            if good_status.active_requests > 0:
                break
            time.sleep(0.01)
        else:
            self.fail("good tier request never became active")

        switched = registry.predict("merchant")
        blocker.set()
        thread.join(timeout=2.0)

        self.assertEqual(switched.tier, "fast")
        self.assertEqual(results, ["good"])


if __name__ == "__main__":
    unittest.main()
