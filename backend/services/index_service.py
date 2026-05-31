"""Index rebuild orchestration."""

from __future__ import annotations

import json
import logging

from backend.db.database import fetch_all
from backend.models.domain import ChunkRecord
from backend.services.ollama_client import OllamaClient
from backend.vectorstore.bm25_store import BM25Store
from backend.vectorstore.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


class IndexService:
    """Rebuild derived retrieval indexes from SQLite chunks."""

    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.faiss_store = FaissVectorStore()
        self.bm25_store = BM25Store()

    async def load_user_chunks(self, user_id: int) -> list[ChunkRecord]:
        """Load all chunks for a user."""

        rows = await fetch_all(
            """
            SELECT id, document_id, user_id, chunk_id, text, token_count, metadata_json
            FROM chunks
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        )
        return [_chunk_from_row(row) for row in rows]

    async def rebuild_user_indexes(self, user_id: int) -> None:
        """Rebuild FAISS and BM25 indexes for one user."""

        chunks = await self.load_user_chunks(user_id)
        await self.bm25_store.build_user_index(user_id, chunks)
        if not chunks:
            await self.faiss_store.build_user_index(user_id, [], [])
            return

        logger.info("Embedding %s chunks for user %s", len(chunks), user_id)
        embeddings = await self.ollama.embed_texts([chunk.text for chunk in chunks])
        await self.faiss_store.build_user_index(user_id, chunks, embeddings)


def _chunk_from_row(row) -> ChunkRecord:
    return ChunkRecord(
        id=int(row["id"]),
        document_id=int(row["document_id"]),
        user_id=int(row["user_id"]),
        chunk_id=str(row["chunk_id"]),
        text=str(row["text"]),
        token_count=int(row["token_count"]),
        metadata=json.loads(row["metadata_json"]),
    )
