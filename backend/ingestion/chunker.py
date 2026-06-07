"""Dependency-free document-aware chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.config.settings import get_settings

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PAGE_MARKER_RE = re.compile(r"^\[Page \d+\]$")


@dataclass(frozen=True)
class TextChunk:
    """A chunk of document text."""

    text: str
    token_count: int


def chunk_text(text: str) -> list[TextChunk]:
    """Split document text into coherent overlapping chunks.

    The strategy is tuned for local RAG over PDFs:
    - keep page markers and paragraphs together when possible;
    - split oversized paragraphs at sentence boundaries;
    - fall back to word windows only for pathological long blocks;
    - use modest overlap to preserve continuity without exploding index size.
    """

    settings = get_settings()
    blocks = _expand_large_blocks(_paragraph_blocks(text), settings.chunk_size_tokens)
    chunks = _pack_blocks(blocks, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    return [TextChunk(text=chunk, token_count=_count_tokens(chunk)) for chunk in chunks]


def _paragraph_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            continue
        if _PAGE_MARKER_RE.match(line):
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            blocks.append(line)
            continue
        current.append(line)

    if current:
        blocks.append(" ".join(current).strip())
    return [block for block in blocks if block]


def _expand_large_blocks(blocks: list[str], chunk_size: int) -> list[str]:
    expanded: list[str] = []
    for block in blocks:
        if _count_tokens(block) <= chunk_size:
            expanded.append(block)
            continue
        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(block) if sentence.strip()]
        if len(sentences) <= 1:
            expanded.extend(_word_windows(block.split(), chunk_size, overlap=0))
        else:
            expanded.extend(_pack_blocks(sentences, chunk_size, overlap=0))
    return expanded


def _pack_blocks(blocks: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = _count_tokens(block)
        if block_tokens > chunk_size:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_tokens = 0
            chunks.extend(_word_windows(block.split(), chunk_size, overlap))
            continue

        if current and current_tokens + block_tokens > chunk_size:
            chunks.append("\n\n".join(current).strip())
            overlap_text = _last_words("\n\n".join(current), overlap)
            current = [overlap_text] if overlap_text else []
            current_tokens = _count_tokens(overlap_text) if overlap_text else 0

        current.append(block)
        current_tokens += block_tokens

    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _word_windows(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _last_words(text: str, count: int) -> str:
    if count <= 0:
        return ""
    return " ".join(text.split()[-count:]).strip()


def _count_tokens(text: str) -> int:
    """Approximate token count cheaply for local chunk sizing."""

    return max(1, len(text.split()))
