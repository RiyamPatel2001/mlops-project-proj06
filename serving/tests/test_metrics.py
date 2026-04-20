from __future__ import annotations

import re
import unittest
from unittest import mock

from app import metrics
from app.layer1 import ModelStatus, RouterSnapshot


def _snapshot() -> RouterSnapshot:
    return RouterSnapshot(
        active_tier="good",
        active_model_name="minilm",
        active_model_version="mlflow-run:good/minilm",
        interactive_tier="good",
        interactive_model_name="minilm",
        bulk_tier="fast",
        bulk_model_name="fasttext",
        pending_tier=None,
        demand_level="normal",
        overload_state="normal",
        active_batch_count=2,
        total_inflight_requests=1,
        last_request_mode="interactive",
        request_rate_rps=1.4,
        models=[
            ModelStatus(
                tier="good",
                model_name="minilm",
                model_kind="hf",
                model_version="mlflow-run:good/minilm",
                ready=True,
                active_requests=1,
                load_error=None,
            ),
            ModelStatus(
                tier="fast",
                model_name="fasttext",
                model_kind="fasttext",
                model_version="mlflow-run:fast/fasttext.bin",
                ready=True,
                active_requests=0,
                load_error=None,
            ),
        ],
    )


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics.reset_metrics_for_tests()
        self.addCleanup(metrics.reset_metrics_for_tests)

    def assert_metric_with_labels(
        self,
        rendered: str,
        metric: str,
        value: str,
        **labels: str,
    ) -> None:
        pattern = re.escape(metric) + r"\{"
        for key, label_value in labels.items():
            pattern += r'[^}]*' + re.escape(f'{key}="{label_value}"')
        pattern += r"[^}]*\}\s+" + re.escape(value)
        self.assertRegex(rendered, pattern)

    @mock.patch("app.metrics.db.pool_available", return_value=True)
    @mock.patch("app.metrics.layer1.get_router_snapshot", return_value=_snapshot())
    def test_classification_metrics_capture_distribution_and_failures(
        self,
        _mock_snapshot,
        _mock_pool,
    ) -> None:
        metrics.record_classification(
            latency=0.12,
            source="layer1",
            request_mode="interactive",
            model_tier="good",
            model_version="mlflow-run:good/minilm",
            confidence=0.58,
            category="Groceries",
        )
        metrics.record_classification(
            latency=0.31,
            source="layer2",
            request_mode="bulk",
            model_tier="layer2",
            model_version="layer2",
            confidence=0.83,
            category="Family Vacation",
        )
        metrics.record_classification_failure(
            latency=1.25,
            request_mode="interactive",
        )

        rendered = metrics.render_metrics().decode("utf-8")

        self.assertIn("serving_requests_total 3.0", rendered)
        self.assert_metric_with_labels(
            rendered,
            "serving_classify_requests_total",
            "1.0",
            source="layer1",
            request_mode="interactive",
            model_tier="good",
            model_version="mlflow-run:good/minilm",
            status="success",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_classify_requests_total",
            "1.0",
            source="unknown",
            request_mode="interactive",
            model_tier="unknown",
            model_version="unknown",
            status="error",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_prediction_outputs_total",
            "1.0",
            source="layer1",
            request_mode="interactive",
            model_tier="good",
            model_version="mlflow-run:good/minilm",
            category_group="Groceries",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_prediction_outputs_total",
            "1.0",
            source="layer2",
            request_mode="bulk",
            model_tier="layer2",
            model_version="layer2",
            category_group="__custom__",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_prediction_confidence_bucket",
            "1.0",
            source="layer1",
            request_mode="interactive",
            model_tier="good",
            model_version="mlflow-run:good/minilm",
            le="0.6",
        )
        self.assertIn("serving_router_request_rate_rps 1.4", rendered)
        self.assertIn("serving_db_connected 1.0", rendered)
        self.assert_metric_with_labels(
            rendered,
            "serving_model_info",
            "1.0",
            tier="good",
            model="minilm",
            kind="hf",
            version="mlflow-run:good/minilm",
        )

    @mock.patch("app.metrics.db.pool_available", return_value=False)
    @mock.patch("app.metrics.layer1.get_router_snapshot", return_value=_snapshot())
    def test_feedback_and_suggestion_metrics_track_review_outcomes(
        self,
        _mock_snapshot,
        _mock_pool,
    ) -> None:
        metrics.record_feedback(
            source="layer1",
            original_prediction="Groceries",
            final_label="Groceries",
            reviewed_by_user=True,
            original_confidence=0.91,
        )
        metrics.record_feedback(
            source="layer1",
            original_prediction="Dining Out",
            final_label="Groceries",
            reviewed_by_user=True,
            original_confidence=0.42,
        )
        metrics.record_feedback_failure(source="layer2")
        metrics.record_suggestion_response(action="accept")
        metrics.record_suggestion_failure(action="dismiss")

        rendered = metrics.render_metrics().decode("utf-8")

        self.assert_metric_with_labels(
            rendered,
            "serving_feedback_total",
            "1.0",
            source="layer1",
            outcome="confirmed",
            status="success",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_feedback_total",
            "1.0",
            source="layer1",
            outcome="corrected",
            status="success",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_feedback_total",
            "1.0",
            source="layer2",
            outcome="error",
            status="error",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_feedback_original_confidence_bucket",
            "1.0",
            source="layer1",
            outcome="corrected",
            le="0.5",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_suggestion_responses_total",
            "1.0",
            action="accept",
            status="success",
        )
        self.assert_metric_with_labels(
            rendered,
            "serving_suggestion_responses_total",
            "1.0",
            action="dismiss",
            status="error",
        )
        self.assertIn("serving_db_connected 0.0", rendered)


if __name__ == "__main__":
    unittest.main()
