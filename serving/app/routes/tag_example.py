"""POST /tag-example — store a user-tagged custom category with its embedding."""

import logging

from fastapi import APIRouter, HTTPException

from app import db, layer2
from app.models import TagExampleRequest, TagExampleResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/tag-example", response_model=TagExampleResponse)
async def tag_example(req: TagExampleRequest) -> TagExampleResponse:
    embedding = await layer2.get_embedding(req.payee)
    if embedding is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable — cannot store example",
        )

    row_id = await db.insert_layer2_example(
        user_id=req.user_id,
        payee=req.payee,
        custom_category=req.custom_category,
        embedding=embedding,
    )
    return TagExampleResponse(status="ok", id=row_id)
