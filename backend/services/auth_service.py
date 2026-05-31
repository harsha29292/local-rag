"""Authentication service and FastAPI dependency."""

from __future__ import annotations

import sqlite3

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.db.database import fetch_one, get_db
from backend.models.domain import User
from backend.utils.security import create_access_token, decode_access_token, hash_password, verify_password

bearer_scheme = HTTPBearer(auto_error=False)


class AuthService:
    """User registration, login, and token validation."""

    async def register(self, username: str, password: str) -> tuple[str, User]:
        """Create a user and return a token."""

        db = await get_db()
        password_hash = hash_password(password)
        try:
            cursor = await db.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username.lower().strip(), password_hash),
            )
            await db.commit()
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists") from exc
        user = User(id=cursor.lastrowid, username=username.lower().strip())
        token = create_access_token(str(user.id), {"username": user.username})
        return token, user

    async def login(self, username: str, password: str) -> tuple[str, User]:
        """Authenticate a user and return a token."""

        db = await get_db()
        row = await fetch_one(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username.lower().strip(),),
        )
        if row is None or not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
        user = User(id=int(row["id"]), username=str(row["username"]))
        token = create_access_token(str(user.id), {"username": user.username})
        return token, user

    async def user_from_token(self, token: str) -> User:
        """Return the current user for a bearer token."""

        try:
            payload = decode_access_token(token)
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc

        user_id = int(payload["sub"])
        db = await get_db()
        row = await fetch_one("SELECT id, username FROM users WHERE id = ?", (user_id,))
        if row is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
        return User(id=int(row["id"]), username=str(row["username"]))


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> User:
    """FastAPI dependency for authenticated routes."""

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return await AuthService().user_from_token(credentials.credentials)
