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
    router_snapshot = layer1.get_router_snapshot()
    return HealthResponse(
        status="healthy" if db.pool_available() else "degraded",
        model_version=router_snapshot.active_model_version,
        active_tier=router_snapshot.active_tier,
        active_model=router_snapshot.active_model_name,
        pending_tier=router_snapshot.pending_tier,
        demand_level=router_snapshot.demand_level,
        request_rate_rps=router_snapshot.request_rate_rps,
        models=[status.__dict__ for status in router_snapshot.models],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    lines: list[str] = []
    router_snapshot = layer1.get_router_snapshot()

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

    lines.append("# HELP serving_router_request_rate_rps Rolling request rate")
    lines.append("# TYPE serving_router_request_rate_rps gauge")
    lines.append(f"serving_router_request_rate_rps {router_snapshot.request_rate_rps:.4f}")

    for demand_level in ("low", "medium", "high"):
        active = 1 if router_snapshot.demand_level == demand_level else 0
        lines.append(
            f'serving_router_demand_level{{level="{demand_level}"}} {active}'
        )

    for tier in ("good", "fast", "cheap"):
        active = 1 if router_snapshot.active_tier == tier else 0
        pending = 1 if router_snapshot.pending_tier == tier else 0
        lines.append(f'serving_router_active_tier{{tier="{tier}"}} {active}')
        lines.append(f'serving_router_pending_tier{{tier="{tier}"}} {pending}')

    for status in router_snapshot.models:
        labels = (
            f'tier="{status.tier}",model="{status.model_name}",'
            f'kind="{status.model_kind}",version="{status.model_version}"'
        )
        lines.append(f"serving_model_ready{{{labels}}} {1 if status.ready else 0}")
        lines.append(f"serving_model_inflight{{{labels}}} {status.active_requests}")
        lines.append(f"serving_model_info{{{labels}}} 1")

    lines.append(f"serving_db_connected {1 if db.pool_available() else 0}")

    return "\n".join(lines) + "\n"
