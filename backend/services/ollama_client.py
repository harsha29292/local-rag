"""Async Ollama HTTP client."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import numpy as np

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
_EMBEDDING_CACHE: dict[str, list[float]] = {}


class OllamaClient:
    """Thin async client for Ollama chat and embedding APIs."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._embedding_cache = _EMBEDDING_CACHE

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON with small retry handling."""

        url = f"{self.settings.ollama_base_url.rstrip('/')}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.settings.ollama_request_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.settings.ollama_timeout_seconds) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.settings.ollama_request_retries:
                    break
                await asyncio.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed: {last_exc}") from last_exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings through Ollama, using a small in-memory cache."""

        if not texts:
            return []

        cached: list[list[float] | None] = [self._embedding_cache.get(text) for text in texts]
        missing_positions = [idx for idx, item in enumerate(cached) if item is None]
        if missing_positions:
            missing_texts = [texts[idx] for idx in missing_positions]
            embeddings = await self._embed_uncached(missing_texts)
            for idx, embedding in zip(missing_positions, embeddings, strict=True):
                normalized = _normalize_vector(embedding)
                cached[idx] = normalized
                self._embedding_cache[texts[idx]] = normalized
        return [item for item in cached if item is not None]

    async def embed_query(self, query: str) -> list[float]:
        """Generate a normalized query embedding."""

        return (await self.embed_texts([query]))[0]

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama embedding endpoints, preferring the batch endpoint."""

        try:
            payload = {"model": self.settings.ollama_embed_model, "input": texts}
            data = await self._post_json("/api/embed", payload)
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                return embeddings
        except Exception as exc:
            logger.debug("Ollama /api/embed unavailable, falling back to /api/embeddings: %s", exc)

        results: list[list[float]] = []
        for text in texts:
            payload = {"model": self.settings.ollama_embed_model, "prompt": text}
            data = await self._post_json("/api/embeddings", payload)
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("Ollama embedding response did not include an embedding")
            results.append(embedding)
        return results

    async def stream_chat(self, messages: list[dict[str, str]], model: str | None = None) -> AsyncIterator[str]:
        """Stream chat completion tokens from Ollama."""

        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model or self.settings.ollama_chat_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_ctx": 4096,
            },
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload, timeout=self.settings.ollama_timeout_seconds) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed Ollama stream line: %s", line)
                        continue
                    if event.get("done"):
                        break
                    content = event.get("message", {}).get("content", "")
                    if content:
                        yield content


def _normalize_vector(vector: list[float]) -> list[float]:
    """Normalize a vector for inner-product cosine search."""

    array = np.asarray(vector, dtype="float32")
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array = array / norm
    return array.astype("float32").tolist()
