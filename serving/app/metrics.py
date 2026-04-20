from __future__ import annotations

import time
from typing import Callable

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.gc_collector import GCCollector
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector

from app import db, layer1
from app.config import LABEL_CLASSES

_CONFIDENCE_BUCKETS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _normalize_request_mode(request_mode: str | None) -> str:
    value = (request_mode or "").strip().lower()
    if value in {"interactive", "bulk"}:
        return value
    return "unknown"


def _normalize_source(source: str | None) -> str:
    value = (source or "").strip().lower()
    if value in {"layer1", "layer2"}:
        return value
    return "unknown"


def _normalize_tier(model_tier: str | None) -> str:
    value = (model_tier or "").strip().lower()
    if value:
        return value
    return "unknown"


def _normalize_version(model_version: str | None) -> str:
    value = (model_version or "").strip()
    if value:
        return value
    return "unknown"


def _normalize_category_group(category: str | None, source: str) -> str:
    value = (category or "").strip()
    if source == "layer2" and value not in LABEL_CLASSES:
        return "__custom__"
    if value in LABEL_CLASSES:
        return value
    return "Other"


def _normalize_feedback_outcome(
    original_prediction: str | None,
    final_label: str | None,
    reviewed_by_user: bool,
) -> str:
    if not reviewed_by_user:
        return "captured"
    original = (original_prediction or "").strip().casefold()
    final = (final_label or "").strip().casefold()
    if not original:
        return "new_label"
    if original == final:
        return "confirmed"
    return "corrected"


def _clamp_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


class _RuntimeMetricsCollector:
    def __init__(self, started_at: Callable[[], float]) -> None:
        self._started_at = started_at

    def describe(self):
        yield GaugeMetricFamily(
            "serving_uptime_seconds",
            "Uptime of the serving process in seconds",
        )
        yield GaugeMetricFamily(
            "serving_router_request_rate_rps",
            "Rolling request rate reported by the Layer 1 router",
        )
        yield GaugeMetricFamily(
            "serving_router_inflight_requests",
            "Current in-flight requests across all Layer 1 runtimes",
        )
        yield GaugeMetricFamily(
            "serving_router_active_batches",
            "Current sticky bulk batches tracked by the router",
        )
        yield GaugeMetricFamily(
            "serving_router_overload_state",
            "Current router overload state",
            labels=["state"],
        )
        yield GaugeMetricFamily(
            "serving_router_default_tier",
            "Default routing tier for each request mode",
            labels=["mode", "tier"],
        )
        yield GaugeMetricFamily(
            "serving_router_last_tier",
            "Tier chosen for the most recently routed request",
            labels=["mode", "tier"],
        )
        yield GaugeMetricFamily(
            "serving_model_ready",
            "Whether a model runtime is loaded and ready",
            labels=["tier", "model", "kind", "version"],
        )
        yield GaugeMetricFamily(
            "serving_model_inflight",
            "In-flight request count for each model runtime",
            labels=["tier", "model", "kind", "version"],
        )
        yield GaugeMetricFamily(
            "serving_model_info",
            "Info metric for model runtimes and versions",
            labels=["tier", "model", "kind", "version"],
        )
        yield GaugeMetricFamily(
            "serving_db_connected",
            "Whether the async Postgres pool is available",
        )

    def collect(self):
        snapshot = layer1.get_router_snapshot()

        uptime = GaugeMetricFamily(
            "serving_uptime_seconds",
            "Uptime of the serving process in seconds",
        )
        uptime.add_metric([], max(0.0, time.time() - self._started_at()))
        yield uptime

        request_rate = GaugeMetricFamily(
            "serving_router_request_rate_rps",
            "Rolling request rate reported by the Layer 1 router",
        )
        request_rate.add_metric([], snapshot.request_rate_rps)
        yield request_rate

        inflight = GaugeMetricFamily(
            "serving_router_inflight_requests",
            "Current in-flight requests across all Layer 1 runtimes",
        )
        inflight.add_metric([], snapshot.total_inflight_requests)
        yield inflight

        batches = GaugeMetricFamily(
            "serving_router_active_batches",
            "Current sticky bulk batches tracked by the router",
        )
        batches.add_metric([], snapshot.active_batch_count)
        yield batches

        overload = GaugeMetricFamily(
            "serving_router_overload_state",
            "Current router overload state",
            labels=["state"],
        )
        for state in ("normal", "warming", "active"):
            overload.add_metric([state], 1 if snapshot.overload_state == state else 0)
        yield overload

        default_tiers = GaugeMetricFamily(
            "serving_router_default_tier",
            "Default routing tier for each request mode",
            labels=["mode", "tier"],
        )
        default_tiers.add_metric(["interactive", snapshot.interactive_tier], 1)
        default_tiers.add_metric(["bulk", snapshot.bulk_tier], 1)
        yield default_tiers

        last_tier = GaugeMetricFamily(
            "serving_router_last_tier",
            "Tier chosen for the most recently routed request",
            labels=["mode", "tier"],
        )
        last_tier.add_metric([snapshot.last_request_mode, snapshot.active_tier], 1)
        yield last_tier

        model_ready = GaugeMetricFamily(
            "serving_model_ready",
            "Whether a model runtime is loaded and ready",
            labels=["tier", "model", "kind", "version"],
        )
        model_inflight = GaugeMetricFamily(
            "serving_model_inflight",
            "In-flight request count for each model runtime",
            labels=["tier", "model", "kind", "version"],
        )
        model_info = GaugeMetricFamily(
            "serving_model_info",
            "Info metric for model runtimes and versions",
            labels=["tier", "model", "kind", "version"],
        )

        for status in snapshot.models:
            labels = [
                status.tier,
                status.model_name,
                status.model_kind,
                status.model_version,
            ]
            model_ready.add_metric(labels, 1 if status.ready else 0)
            model_inflight.add_metric(labels, status.active_requests)
            model_info.add_metric(labels, 1)

        yield model_ready
        yield model_inflight
        yield model_info

        db_connected = GaugeMetricFamily(
            "serving_db_connected",
            "Whether the async Postgres pool is available",
        )
        db_connected.add_metric([], 1 if db.pool_available() else 0)
        yield db_connected


class MetricsStore:
    def __init__(self) -> None:
        self._started_at = time.time()
        self.registry = CollectorRegistry()

        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        GCCollector(registry=self.registry)
        self.registry.register(_RuntimeMetricsCollector(self.started_at))

        self.requests_total = Counter(
            "serving_requests_total",
            "Total classify requests",
            registry=self.registry,
        )
        self.classify_requests = Counter(
            "serving_classify_requests_total",
            "Total classify requests by route outcome and routing labels",
            labelnames=["source", "request_mode", "model_tier", "model_version", "status"],
            registry=self.registry,
        )
        self.classify_latency = Histogram(
            "serving_classify_request_latency_seconds",
            "Latency of classify requests",
            labelnames=["source", "request_mode", "model_tier", "status"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.prediction_outputs = Counter(
            "serving_prediction_outputs_total",
            "Returned prediction outputs grouped to a bounded label set",
            labelnames=["source", "request_mode", "model_tier", "model_version", "category_group"],
            registry=self.registry,
        )
        self.prediction_confidence = Histogram(
            "serving_prediction_confidence",
            "Confidence distribution of returned predictions",
            labelnames=["source", "request_mode", "model_tier", "model_version"],
            buckets=_CONFIDENCE_BUCKETS,
            registry=self.registry,
        )
        self.feedback_total = Counter(
            "serving_feedback_total",
            "Feedback submissions by source and review outcome",
            labelnames=["source", "outcome", "status"],
            registry=self.registry,
        )
        self.feedback_original_confidence = Histogram(
            "serving_feedback_original_confidence",
            "Original confidence values attached to feedback submissions",
            labelnames=["source", "outcome"],
            buckets=_CONFIDENCE_BUCKETS,
            registry=self.registry,
        )
        self.suggestion_responses = Counter(
            "serving_suggestion_responses_total",
            "User responses to category suggestions",
            labelnames=["action", "status"],
            registry=self.registry,
        )

    def started_at(self) -> float:
        return self._started_at

    def record_classification(
        self,
        *,
        latency: float,
        source: str,
        request_mode: str,
        model_tier: str,
        model_version: str,
        confidence: float | None,
        category: str | None,
    ) -> None:
        normalized_source = _normalize_source(source)
        normalized_mode = _normalize_request_mode(request_mode)
        normalized_tier = _normalize_tier(model_tier)
        normalized_version = _normalize_version(model_version)

        self.requests_total.inc()
        self.classify_requests.labels(
            source=normalized_source,
            request_mode=normalized_mode,
            model_tier=normalized_tier,
            model_version=normalized_version,
            status="success",
        ).inc()
        self.classify_latency.labels(
            source=normalized_source,
            request_mode=normalized_mode,
            model_tier=normalized_tier,
            status="success",
        ).observe(max(0.0, float(latency)))
        self.prediction_outputs.labels(
            source=normalized_source,
            request_mode=normalized_mode,
            model_tier=normalized_tier,
            model_version=normalized_version,
            category_group=_normalize_category_group(category, normalized_source),
        ).inc()

        normalized_confidence = _clamp_confidence(confidence)
        if normalized_confidence is not None:
            self.prediction_confidence.labels(
                source=normalized_source,
                request_mode=normalized_mode,
                model_tier=normalized_tier,
                model_version=normalized_version,
            ).observe(normalized_confidence)

    def record_classification_failure(
        self,
        *,
        latency: float,
        request_mode: str | None,
    ) -> None:
        normalized_mode = _normalize_request_mode(request_mode)
        self.requests_total.inc()
        self.classify_requests.labels(
            source="unknown",
            request_mode=normalized_mode,
            model_tier="unknown",
            model_version="unknown",
            status="error",
        ).inc()
        self.classify_latency.labels(
            source="unknown",
            request_mode=normalized_mode,
            model_tier="unknown",
            status="error",
        ).observe(max(0.0, float(latency)))

    def record_feedback(
        self,
        *,
        source: str,
        original_prediction: str | None,
        final_label: str | None,
        reviewed_by_user: bool,
        original_confidence: float | None,
    ) -> None:
        normalized_source = _normalize_source(source)
        outcome = _normalize_feedback_outcome(
            original_prediction=original_prediction,
            final_label=final_label,
            reviewed_by_user=reviewed_by_user,
        )
        self.feedback_total.labels(
            source=normalized_source,
            outcome=outcome,
            status="success",
        ).inc()

        normalized_confidence = _clamp_confidence(original_confidence)
        if normalized_confidence is not None:
            self.feedback_original_confidence.labels(
                source=normalized_source,
                outcome=outcome,
            ).observe(normalized_confidence)

    def record_feedback_failure(self, *, source: str) -> None:
        self.feedback_total.labels(
            source=_normalize_source(source),
            outcome="error",
            status="error",
        ).inc()

    def record_suggestion_response(self, *, action: str) -> None:
        normalized_action = (action or "").strip().lower() or "unknown"
        self.suggestion_responses.labels(
            action=normalized_action,
            status="success",
        ).inc()

    def record_suggestion_failure(self, *, action: str | None) -> None:
        normalized_action = (action or "").strip().lower() or "unknown"
        self.suggestion_responses.labels(
            action=normalized_action,
            status="error",
        ).inc()

    def render_latest(self) -> bytes:
        return generate_latest(self.registry)


_STORE = MetricsStore()


def record_classification(
    *,
    latency: float,
    source: str,
    request_mode: str,
    model_tier: str,
    model_version: str,
    confidence: float | None,
    category: str | None,
) -> None:
    _STORE.record_classification(
        latency=latency,
        source=source,
        request_mode=request_mode,
        model_tier=model_tier,
        model_version=model_version,
        confidence=confidence,
        category=category,
    )


def record_classification_failure(
    *,
    latency: float,
    request_mode: str | None,
) -> None:
    _STORE.record_classification_failure(
        latency=latency,
        request_mode=request_mode,
    )


def record_feedback(
    *,
    source: str,
    original_prediction: str | None,
    final_label: str | None,
    reviewed_by_user: bool,
    original_confidence: float | None,
) -> None:
    _STORE.record_feedback(
        source=source,
        original_prediction=original_prediction,
        final_label=final_label,
        reviewed_by_user=reviewed_by_user,
        original_confidence=original_confidence,
    )


def record_feedback_failure(*, source: str) -> None:
    _STORE.record_feedback_failure(source=source)


def record_suggestion_response(*, action: str) -> None:
    _STORE.record_suggestion_response(action=action)


def record_suggestion_failure(*, action: str | None) -> None:
    _STORE.record_suggestion_failure(action=action)


def render_metrics() -> bytes:
    return _STORE.render_latest()


def reset_metrics_for_tests() -> None:
    global _STORE
    _STORE = MetricsStore()
