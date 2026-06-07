"""Authentication request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthRequest(BaseModel):
    """Register or login payload."""

    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=72)
    registration_code: str | None = Field(default=None, max_length=128)


class TokenResponse(BaseModel):
    """Bearer token response."""

    access_token: str
    token_type: str = "bearer"
    username: str


class UserResponse(BaseModel):
    """Current user response."""

    id: int
    username: str
