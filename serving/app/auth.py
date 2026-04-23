from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import Header, HTTPException, status

from app import db
from app.config import AUTH_PASSWORD_HASH_ITERATIONS, AUTH_SESSION_TTL_HOURS


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    username: str


def normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if not normalized:
        raise ValueError('invalid-username')
    return normalized


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        AUTH_PASSWORD_HASH_ITERATIONS,
    )
    return (
        f'pbkdf2_sha256${AUTH_PASSWORD_HASH_ITERATIONS}'
        f'${salt}${digest.hex()}'
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt, digest_hex = stored_hash.split('$', 3)
    except ValueError:
        return hmac.compare_digest(password, stored_hash)

    if algorithm != 'pbkdf2_sha256':
        return False

    digest = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        int(iterations),
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def create_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def get_session_expiry() -> datetime:
    return datetime.utcnow() + timedelta(hours=AUTH_SESSION_TTL_HOURS)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={'WWW-Authenticate': 'Bearer'},
    )


async def require_authenticated_user(
    authorization: Annotated[str | None, Header(alias='Authorization')] = None,
) -> AuthenticatedUser:
    if not authorization:
        raise _unauthorized('missing-auth-token')

    scheme, _, token = authorization.partition(' ')
    if scheme.lower() != 'bearer' or not token:
        raise _unauthorized('invalid-auth-token')

    user = await db.get_auth_user_for_session(hash_session_token(token))
    if not user:
        raise _unauthorized('invalid-auth-token')

    return AuthenticatedUser(
        user_id=user['user_id'],
        username=user['username'],
    )
