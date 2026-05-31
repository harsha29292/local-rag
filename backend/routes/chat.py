"""General chat and conversation routes."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from backend.models.domain import User
from backend.schemas.chat import ChatRequest, ConversationCreate, ConversationResponse, MessageResponse
from backend.services.auth_service import get_current_user
from backend.services.chat_service import ChatService
from backend.services.memory_service import MemoryService

router = APIRouter()


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    mode: str | None = Query(default=None, pattern="^(general|rag)$"),
    current_user: User = Depends(get_current_user),
) -> list[ConversationResponse]:
    """List conversations for the authenticated user."""

    return await MemoryService().list_conversations(current_user, mode)


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
) -> ConversationResponse:
    """Create a conversation."""

    return await MemoryService().create_conversation(current_user, payload.title, payload.mode)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def list_messages(conversation_id: int, current_user: User = Depends(get_current_user)) -> list[MessageResponse]:
    """List messages for one conversation."""

    return await MemoryService().list_messages(current_user, conversation_id)


@router.post("/general/stream")
async def stream_general_chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream a general LLM chat response as NDJSON."""

    return StreamingResponse(
        _jsonl(ChatService().stream_general_chat(current_user, payload)),
        media_type="application/x-ndjson",
    )


async def _jsonl(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for event in events:
        yield json.dumps(event, ensure_ascii=False) + "\n"
