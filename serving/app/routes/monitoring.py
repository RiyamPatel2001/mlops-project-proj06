"""GET /health  +  GET /metrics"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response

from app import db, layer1
from app.metrics import render_metrics
from app.models import HealthResponse

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

router = APIRouter()

_start_time: float = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    router_snapshot = layer1.get_router_snapshot()
    return HealthResponse(
        status="healthy" if db.pool_available() else "degraded",
        model_version=router_snapshot.active_model_version,
        active_tier=router_snapshot.active_tier,
        active_model=router_snapshot.active_model_name,
        interactive_tier=router_snapshot.interactive_tier,
        interactive_model=router_snapshot.interactive_model_name,
        bulk_tier=router_snapshot.bulk_tier,
        bulk_model=router_snapshot.bulk_model_name,
        pending_tier=router_snapshot.pending_tier,
        demand_level=router_snapshot.demand_level,
        overload_state=router_snapshot.overload_state,
        active_batch_count=router_snapshot.active_batch_count,
        total_inflight_requests=router_snapshot.total_inflight_requests,
        last_request_mode=router_snapshot.last_request_mode,
        request_rate_rps=router_snapshot.request_rate_rps,
        models=[status.__dict__ for status in router_snapshot.models],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(
        content=render_metrics(),
        media_type=PROMETHEUS_CONTENT_TYPE,
    )
