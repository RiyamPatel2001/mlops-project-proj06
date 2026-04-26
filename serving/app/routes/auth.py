from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app import db
from app.auth import (
    AuthenticatedUser,
    create_session_token,
    get_session_expiry,
    hash_password,
    hash_session_token,
    normalize_username,
    require_authenticated_user,
    verify_password,
)
from app.models import (
    AuthLoginRequest,
    AuthLoginResponse,
    AuthMeResponse,
    AuthRegisterRequest,
    AuthRegisterResponse,
    StatusResponse,
)

router = APIRouter()


@router.post(
    '/auth/register',
    response_model=AuthRegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(req: AuthRegisterRequest) -> AuthRegisterResponse:
    try:
        normalized_username = normalize_username(req.username)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    user = await db.create_auth_user(
        user_id=str(uuid.uuid4()),
        username=req.username.strip(),
        username_normalized=normalized_username,
        password_hash=hash_password(req.password),
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='username-taken',
        )

    return AuthRegisterResponse(
        user_id=user['user_id'],
        username=user['username'],
    )


@router.post('/auth/login', response_model=AuthLoginResponse)
async def login_user(req: AuthLoginRequest) -> AuthLoginResponse:
    try:
        normalized_username = normalize_username(req.username)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    user = await db.get_auth_user_by_username(normalized_username)
    if not user or not verify_password(req.password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='invalid-credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    token = create_session_token()
    await db.create_auth_session(
        user_id=user['user_id'],
        token_hash=hash_session_token(token),
        expires_at=get_session_expiry(),
    )

    return AuthLoginResponse(
        user_id=user['user_id'],
        username=user['username'],
        token=token,
    )


@router.get('/auth/me', response_model=AuthMeResponse)
async def get_current_user(
    current_user: AuthenticatedUser = Depends(require_authenticated_user),
) -> AuthMeResponse:
    return AuthMeResponse(
        user_id=current_user.user_id,
        username=current_user.username,
    )


@router.post('/auth/logout', response_model=StatusResponse)
async def logout_user(
    current_user: AuthenticatedUser = Depends(require_authenticated_user),
    authorization: str | None = Header(default=None, alias='Authorization'),
) -> StatusResponse:
    del current_user
    if not authorization:
        return StatusResponse(status='ok')

    _, _, token = authorization.partition(' ')
    if token:
        await db.delete_auth_session(hash_session_token(token))
    return StatusResponse(status='ok')
