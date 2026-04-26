from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Annotated, Optional

from fastapi import Header, HTTPException, Request, status

ACTUAL_USER_ID_HEADER = 'X-Actual-User-Id'
ACTUAL_USERNAME_HEADER = 'X-Actual-Username'
ACTUAL_TIMESTAMP_HEADER = 'X-Actual-Auth-Timestamp'
ACTUAL_SIGNATURE_HEADER = 'X-Actual-Auth-Signature'
DEFAULT_ACTUAL_AUTH_MAX_AGE_SECONDS = 30


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    username: str


def _shared_secret() -> str:
    return os.getenv('ACTUAL_ML_SHARED_SECRET', '')


def _max_auth_age_seconds() -> int:
    raw_value = os.getenv(
        'ACTUAL_ML_AUTH_MAX_AGE_SECONDS',
        str(DEFAULT_ACTUAL_AUTH_MAX_AGE_SECONDS),
    )
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_ACTUAL_AUTH_MAX_AGE_SECONDS


def build_actual_proxy_signature(
    *,
    secret: str,
    user_id: str,
    username: str,
    method: str,
    path: str,
    query: str,
    timestamp: str,
) -> str:
    payload = '\n'.join(
        [user_id, username, method.upper(), path, query, timestamp],
    )
    return hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256,
    ).hexdigest()


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
    )


async def require_authenticated_user(
    request: Request,
    actual_user_id: Annotated[
        Optional[str], Header(alias=ACTUAL_USER_ID_HEADER)
    ] = None,
    actual_username: Annotated[
        Optional[str], Header(alias=ACTUAL_USERNAME_HEADER)
    ] = None,
    actual_auth_timestamp: Annotated[
        Optional[str], Header(alias=ACTUAL_TIMESTAMP_HEADER)
    ] = None,
    actual_auth_signature: Annotated[
        Optional[str], Header(alias=ACTUAL_SIGNATURE_HEADER)
    ] = None,
) -> AuthenticatedUser:
    secret = _shared_secret()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='actual-auth-not-configured',
        )

    if not (
        actual_user_id
        and actual_username
        and actual_auth_timestamp
        and actual_auth_signature
    ):
        raise _unauthorized('missing-actual-auth')

    try:
        timestamp = int(actual_auth_timestamp)
    except ValueError as exc:
        raise _unauthorized('invalid-actual-auth') from exc

    if abs(int(time.time()) - timestamp) > _max_auth_age_seconds():
        raise _unauthorized('stale-actual-auth')

    expected_signature = build_actual_proxy_signature(
        secret=secret,
        user_id=actual_user_id,
        username=actual_username,
        method=request.method,
        path=request.url.path,
        query=request.url.query,
        timestamp=actual_auth_timestamp,
    )
    if not hmac.compare_digest(expected_signature, actual_auth_signature):
        raise _unauthorized('invalid-actual-auth')

    return AuthenticatedUser(
        user_id=actual_user_id,
        username=actual_username,
    )
