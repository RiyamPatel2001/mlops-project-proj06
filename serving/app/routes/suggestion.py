"""POST /suggestion-response — user accepts or dismisses a suggestion."""

from fastapi import APIRouter

from app import db
from app.models import StatusResponse, SuggestionResponseRequest

router = APIRouter()


@router.post("/suggestion-response", response_model=StatusResponse)
async def suggestion_response(req: SuggestionResponseRequest) -> StatusResponse:
    await db.insert_suggestion_response(
        user_id=req.user_id,
        transaction_id=req.transaction_id,
        action=req.action.value,
        suggested_category=req.suggested_category,
    )
    return StatusResponse(status="ok")
