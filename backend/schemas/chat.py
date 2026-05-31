"""Chat schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    """Create conversation payload."""

    title: str = Field(default="New conversation", max_length=120)
    mode: Literal["general", "rag"] = "general"


class ConversationResponse(BaseModel):
    """Conversation list item."""

    id: int
    title: str
    mode: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    """Message returned to the UI."""

    id: int
    conversation_id: int
    role: str
    content: str
    metadata: dict
    created_at: str


class ChatRequest(BaseModel):
    """General chat request."""

    message: str = Field(min_length=1, max_length=8000)
    conversation_id: int | None = None
