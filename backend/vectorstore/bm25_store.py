"""Per-user BM25 sparse retrieval store."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.config.settings import get_settings
from backend.models.domain import ChunkRecord, RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class BM25Store:
    """BM25 sparse index persisted as tokenized chunk records."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _user_dir(self, user_id: int) -> Path:
        return self.settings.bm25_dir / f"user_{user_id}"

    def _index_path(self, user_id: int) -> Path:
        return self._user_dir(user_id) / "bm25.json"

    async def build_user_index(self, user_id: int, chunks: list[ChunkRecord]) -> None:
        """Persist tokenized corpus for a user's BM25 index."""

        await asyncio.to_thread(self._build_user_index_sync, user_id, chunks)

    def _build_user_index_sync(self, user_id: int, chunks: list[ChunkRecord]) -> None:
        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = self._index_path(user_id)
        if not chunks:
            path.unlink(missing_ok=True)
            return
        payload = {
            "chunks": [_chunk_to_json(chunk) for chunk in chunks],
            "tokens": [_tokenize(chunk.text) for chunk in chunks],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    async def search(self, user_id: int, query: str, top_k: int) -> list[RetrievedChunk]:
        """Search a user's BM25 index."""

        return await asyncio.to_thread(self._search_sync, user_id, query, top_k)

    def _search_sync(self, user_id: int, query: str, top_k: int) -> list[RetrievedChunk]:
        path = self._index_path(user_id)
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens: list[list[str]] = payload.get("tokens", [])
        if not tokens:
            return []

        bm25 = BM25Okapi(tokens)
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:top_k]

        chunks = payload.get("chunks", [])
        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if float(score) <= 0:
                continue
            chunk = _chunk_from_json(chunks[idx])
            results.append(RetrievedChunk(chunk=chunk, score=float(score), source="sparse", ranks={"sparse": rank}))
        return results


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _chunk_to_json(chunk: ChunkRecord) -> dict:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "user_id": chunk.user_id,
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata,
    }


def _chunk_from_json(data: dict) -> ChunkRecord:
    return ChunkRecord(
        id=int(data["id"]),
        document_id=int(data["document_id"]),
        user_id=int(data["user_id"]),
        chunk_id=str(data["chunk_id"]),
        text=str(data["text"]),
        token_count=int(data["token_count"]),
        metadata=dict(data.get("metadata", {})),
    )
