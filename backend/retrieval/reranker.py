"""Lightweight reranking layer."""

from __future__ import annotations

import asyncio
import logging
import re

from backend.config.settings import get_settings
from backend.models.domain import RetrievedChunk

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class Reranker:
    """Optional MiniLM cross-encoder reranker with lexical fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._load_failed = False

    async def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Rerank candidate chunks."""

        if not candidates:
            return []
        max_candidates = self.settings.reranker_max_candidates
        candidates = candidates[:max_candidates]
        if self.settings.reranker_enabled and not self._load_failed:
            try:
                return await asyncio.to_thread(self._cross_encoder_rerank, query, candidates)
            except Exception as exc:
                self._load_failed = True
                logger.warning("Cross-encoder reranker unavailable, using lexical fallback: %s", exc)
        return self._lexical_rerank(query, candidates)

    def _cross_encoder_rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.reranker_model)
        pairs = [(query, item.chunk.text) for item in candidates]
        scores = self._model.predict(pairs)
        reranked: list[RetrievedChunk] = []
        for item, score in zip(candidates, scores, strict=True):
            item.score = float(score)
            item.source = "rerank"
            reranked.append(item)
        return sorted(reranked, key=lambda item: item.score, reverse=True)

    def _lexical_rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_tokens = set(_tokenize(query))
        reranked: list[RetrievedChunk] = []
        for item in candidates:
            text_tokens = set(_tokenize(item.chunk.text))
            overlap = len(query_tokens & text_tokens)
            item.score = item.score + (overlap / max(len(query_tokens), 1))
            item.source = "rerank"
            reranked.append(item)
        return sorted(reranked, key=lambda item: item.score, reverse=True)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
