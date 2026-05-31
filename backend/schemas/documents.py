"""Document schemas."""

from __future__ import annotations

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """Document metadata returned by the API."""

    id: int
    filename: str
    original_filename: str
    status: str
    chunk_count: int
    error_message: str | None = None
    created_at: str
    updated_at: str


class DocumentIngestResponse(BaseModel):
    """Document ingestion result."""

    document: DocumentResponse
    message: str
