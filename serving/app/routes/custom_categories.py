"""GET /custom-categories — return a user's custom categories."""

from fastapi import APIRouter, Query

from app import db
from app.models import CustomCategoriesResponse

router = APIRouter()


@router.get("/custom-categories", response_model=CustomCategoriesResponse)
async def get_custom_categories(
    user_id: str = Query(..., description="User / account ID"),
) -> CustomCategoriesResponse:
    cats = await db.get_user_custom_categories(user_id)
    return CustomCategoriesResponse(categories=cats)
