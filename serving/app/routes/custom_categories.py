"""GET /custom-categories — return a user's custom categories."""

from fastapi import APIRouter, Depends

from app import db
from app.auth import AuthenticatedUser, require_authenticated_user
from app.models import CustomCategoriesResponse

router = APIRouter()


@router.get("/custom-categories", response_model=CustomCategoriesResponse)
async def get_custom_categories(
    current_user: AuthenticatedUser = Depends(require_authenticated_user),
) -> CustomCategoriesResponse:
    cats = await db.get_user_custom_categories(current_user.user_id)
    return CustomCategoriesResponse(categories=cats)
