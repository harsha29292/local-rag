"""Hybrid retrieval service."""

from __future__ import annotations

import asyncio
import logging

from backend.config.settings import get_settings
from backend.models.domain import RetrievedChunk, User
from backend.retrieval.reranker import Reranker
from backend.schemas.rag import SourceChunk
from backend.services.ollama_client import OllamaClient
from backend.vectorstore.bm25_store import BM25Store
from backend.vectorstore.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    """Dense + sparse hybrid retrieval with RRF and reranking."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.ollama = OllamaClient()
        self.faiss_store = FaissVectorStore()
        self.bm25_store = BM25Store()
        self.reranker = Reranker()

    async def retrieve(self, user: User, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Run hybrid retrieval scoped to one user."""

        query = _normalize_query(query)
        final_top_k = top_k or self.settings.retrieval_final_top_k
        sparse_task = asyncio.create_task(self.bm25_store.search(user.id, query, self.settings.retrieval_sparse_top_k))

        dense_results: list[RetrievedChunk] = []
        try:
            query_embedding = await self.ollama.embed_query(query)
            dense_results = await self.faiss_store.search(user.id, query_embedding, self.settings.retrieval_dense_top_k)
        except Exception as exc:
            logger.warning("Dense retrieval unavailable; continuing with sparse results: %s", exc)

        sparse_results = await sparse_task
        fused = self._reciprocal_rank_fusion([dense_results, sparse_results])
        reranked = await self.reranker.rerank(query, fused)
        return reranked[:final_top_k]

    def _reciprocal_rank_fusion(self, rankings: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        combined: dict[int, RetrievedChunk] = {}
        scores: dict[int, float] = {}

        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                chunk_id = item.chunk.id
                if chunk_id not in combined:
                    combined[chunk_id] = RetrievedChunk(chunk=item.chunk, score=0.0, source="hybrid", ranks={})
                    scores[chunk_id] = 0.0
                source_name = item.source
                combined[chunk_id].ranks[source_name] = rank
                scores[chunk_id] += 1.0 / (self.settings.rrf_k + rank)

        for chunk_id, score in scores.items():
            combined[chunk_id].score = score
        return sorted(combined.values(), key=lambda item: item.score, reverse=True)


def source_chunks(results: list[RetrievedChunk]) -> list[SourceChunk]:
    """Convert retrieved chunks to API source payloads."""

    return [
        SourceChunk(
            chunk_id=item.chunk.chunk_id,
            document_id=item.chunk.document_id,
            filename=str(item.chunk.metadata.get("filename", "document")),
            text=item.chunk.text,
            score=item.score,
            source=item.source,
        )
        for item in results
    ]


def _normalize_query(query: str) -> str:
    """Clean small common input slips before retrieval."""

    cleaned = " ".join(query.strip().split())
    lower = cleaned.lower()
    if lower.startswith("hich "):
        cleaned = "w" + cleaned
    return cleaned
