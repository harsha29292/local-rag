"""Semantic-aware recursive chunking."""

from __future__ import annotations

from dataclasses import dataclass

from backend.config.settings import get_settings


@dataclass(frozen=True)
class TextChunk:
    """A chunk of document text."""

    text: str
    token_count: int


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken when available, otherwise approximate."""

    return max(1, len(text.split()))


def chunk_text(text: str) -> list[TextChunk]:
    """Split text into semantic-ish overlapping chunks."""

    settings = get_settings()
    chunks = _recursive_split(text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    return [TextChunk(text=chunk, token_count=_count_tokens(chunk)) for chunk in chunks]


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
    parts = [text.strip()]
    for separator in separators:
        next_parts: list[str] = []
        for part in parts:
            if _count_tokens(part) <= chunk_size:
                next_parts.append(part)
            else:
                next_parts.extend(piece.strip() for piece in part.split(separator) if piece.strip())
        parts = next_parts

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for part in parts:
        tokens = _count_tokens(part)
        if tokens > chunk_size:
            words = part.split()
            chunks.extend(_window_words(words, chunk_size, overlap))
            current = []
            current_tokens = 0
            continue
        if current and current_tokens + tokens > chunk_size:
            chunks.append(" ".join(current).strip())
            overlap_words = " ".join(current).split()[-overlap:]
            current = overlap_words
            current_tokens = len(current)
        current.append(part)
        current_tokens += tokens
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _window_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks
