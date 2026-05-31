"""Security helpers for auth and prompt construction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import bcrypt

from backend.config.settings import get_settings


def hash_password(password: str) -> str:
    """Hash a plaintext password."""

    password_bytes = password.encode("utf-8")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a hash."""

    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    """Create a JWT access token."""

    settings = get_settings()
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {"sub": subject, "exp": expires}
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT."""

    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def strip_control_chars(text: str) -> str:
    """Remove most control characters from text."""

    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
