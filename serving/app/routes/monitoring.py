"""GET /health  +  GET /metrics"""

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app import db, layer1
from app.models import HealthResponse

router = APIRouter()

_request_count: int = 0
_request_latencies: list[float] = []
_confidence_values: list[float] = []

_start_time: float = time.time()


def record_request(latency: float, confidence: float | None = None) -> None:
    global _request_count
    _request_count += 1
    _request_latencies.append(latency)
    if len(_request_latencies) > 10_000:
        _request_latencies.pop(0)
    if confidence is not None:
        _confidence_values.append(confidence)
        if len(_confidence_values) > 10_000:
            _confidence_values.pop(0)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if db.pool_available() else "degraded",
        model_version=layer1.get_model_version(),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    lines: list[str] = []

    lines.append("# HELP serving_requests_total Total classify requests")
    lines.append("# TYPE serving_requests_total counter")
    lines.append(f"serving_requests_total {_request_count}")

    if _request_latencies:
        avg_lat = sum(_request_latencies) / len(_request_latencies)
        lines.append("# HELP serving_latency_seconds Average request latency")
        lines.append("# TYPE serving_latency_seconds gauge")
        lines.append(f"serving_latency_seconds {avg_lat:.6f}")

    if _confidence_values:
        avg_conf = sum(_confidence_values) / len(_confidence_values)
        lines.append("# HELP serving_confidence_avg Average prediction confidence")
        lines.append("# TYPE serving_confidence_avg gauge")
        lines.append(f"serving_confidence_avg {avg_conf:.4f}")

        for bucket in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            count = sum(1 for c in _confidence_values if c <= bucket)
            lines.append(f'serving_confidence_bucket{{le="{bucket}"}} {count}')
        lines.append(
            f"serving_confidence_bucket{{le=\"+Inf\"}} {len(_confidence_values)}"
        )

    lines.append(f'serving_model_info{{version="{layer1.get_model_version()}"}} 1')
    lines.append(f"serving_db_connected {1 if db.pool_available() else 0}")

    return "\n".join(lines) + "\n"
