"""RAG schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """Retrieved source returned to clients."""

    chunk_id: str
    document_id: int
    filename: str
    text: str
    score: float
    source: str


class RagRequest(BaseModel):
    """RAG query request."""

    question: str = Field(min_length=1, max_length=8000)
    conversation_id: int | None = None
    top_k: int | None = Field(default=None, ge=1, le=12)
