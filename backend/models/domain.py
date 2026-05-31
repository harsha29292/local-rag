"""Typed domain objects used across services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class User:
    """Authenticated user."""

    id: int
    username: str


@dataclass(frozen=True)
class ChunkRecord:
    """Stored document chunk."""

    id: int
    document_id: int
    user_id: int
    chunk_id: str
    text: str
    token_count: int
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    """Chunk returned from retrieval."""

    chunk: ChunkRecord
    score: float
    source: Literal["dense", "sparse", "hybrid", "rerank"] = "hybrid"
    ranks: dict[str, int] = field(default_factory=dict)
