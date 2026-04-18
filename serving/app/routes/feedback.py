"""POST /feedback  +  GET /feedback/export"""

import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter

from app import db
from app.models import FeedbackExportRow, FeedbackRequest, StatusResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=StatusResponse)
async def submit_feedback(req: FeedbackRequest) -> StatusResponse:
    if req.timestamp:
        raw_ts = req.timestamp.replace("Z", "+00:00")
        parsed_ts = datetime.fromisoformat(raw_ts)
        ts = (
            parsed_ts.astimezone(timezone.utc).replace(tzinfo=None)
            if parsed_ts.tzinfo is not None
            else parsed_ts
        )
    else:
        ts = datetime.utcnow()

    try:
        parsed_date = date.fromisoformat(req.date)
    except ValueError:
        parsed_date = datetime.strptime(req.date, "%Y-%m-%d").date()

    row = {
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "payee": req.payee,
        "amount": req.amount,
        "date": parsed_date,
        "original_prediction": req.original_prediction,
        "original_confidence": req.original_confidence,
        "source": req.source,
        "final_label": req.final_label,
        "reviewed_by_user": req.reviewed_by_user,
        "timestamp": ts,
    }
    await db.insert_feedback(row)
    return StatusResponse(status="ok")


@router.get("/feedback/export", response_model=list[FeedbackExportRow])
async def export_feedback() -> list[FeedbackExportRow]:
    """Expose reviewed layer-1 feedback for Saketh's batch pipeline."""
    return await db.export_feedback()
