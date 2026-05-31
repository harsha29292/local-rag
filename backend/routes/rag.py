"""RAG query routes."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.models.domain import User
from backend.schemas.rag import RagRequest
from backend.services.auth_service import get_current_user
from backend.services.rag_service import RagService

router = APIRouter()


@router.post("/query/stream")
async def stream_rag_query(
    payload: RagRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream a RAG answer as NDJSON."""

    return StreamingResponse(
        _jsonl(RagService().stream_rag_answer(current_user, payload)),
        media_type="application/x-ndjson",
    )


async def _jsonl(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for event in events:
        yield json.dumps(event, ensure_ascii=False) + "\n"
