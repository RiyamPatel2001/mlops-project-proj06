"""POST /suggestion-response — user accepts or dismisses a suggestion."""

import logging

from fastapi import APIRouter

from app import db
from app.metrics import record_suggestion_failure, record_suggestion_response
from app.models import StatusResponse, SuggestionResponseRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/suggestion-response", response_model=StatusResponse)
async def suggestion_response(req: SuggestionResponseRequest) -> StatusResponse:
    try:
        await db.insert_suggestion_response(
            user_id=req.user_id,
            transaction_id=req.transaction_id,
            action=req.action.value,
            suggested_category=req.suggested_category,
        )
    except Exception:
        logger.exception(
            "Suggestion response insert failed for transaction_id=%s user_id=%s",
            req.transaction_id,
            req.user_id,
        )
        record_suggestion_failure(action=req.action.value)
        raise

    record_suggestion_response(action=req.action.value)
    return StatusResponse(status="ok")
