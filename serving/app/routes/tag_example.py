"""POST /tag-example — store a user-tagged custom category with its embedding."""

import logging

from fastapi import APIRouter, Depends

from app import db, layer2
from app.auth import AuthenticatedUser, require_authenticated_user
from app.feature_computation import (
    bin_amount,
    day_of_month,
    day_of_week,
    normalize_payee,
)
from app.models import TagExampleRequest, TagExampleResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/tag-example", response_model=TagExampleResponse)
async def tag_example(
    req: TagExampleRequest,
    current_user: AuthenticatedUser = Depends(require_authenticated_user),
) -> TagExampleResponse:
    text_parts = [normalize_payee(req.payee)]
    if req.amount is not None:
        text_parts.append(bin_amount(abs(req.amount)))
    if req.date:
        text_parts.append(day_of_week(req.date))
        text_parts.append(str(day_of_month(req.date)))

    text = " ".join(text_parts)
    embedding = await layer2.get_embedding(text)

    row_id = await db.insert_layer2_example(
        user_id=current_user.user_id,
        payee=req.payee,
        custom_category=req.custom_category,
        embedding=embedding,
    )
    return TagExampleResponse(status="ok", id=row_id)
