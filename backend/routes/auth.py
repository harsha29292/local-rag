"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.models.domain import User
from backend.schemas.auth import AuthRequest, TokenResponse, UserResponse
from backend.services.auth_service import AuthService, get_current_user

router = APIRouter()


@router.post("/register", response_model=TokenResponse)
async def register(payload: AuthRequest) -> TokenResponse:
    """Register a new local user."""

    token, user = await AuthService().register(payload.username, payload.password, payload.registration_code)
    return TokenResponse(access_token=token, username=user.username)


@router.post("/login", response_model=TokenResponse)
async def login(payload: AuthRequest) -> TokenResponse:
    """Login and return a bearer token."""

    token, user = await AuthService().login(payload.username, payload.password)
    return TokenResponse(access_token=token, username=user.username)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return the authenticated user."""

    return UserResponse(id=current_user.id, username=current_user.username)
