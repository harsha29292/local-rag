"""Prompt builder tests."""

from __future__ import annotations

from backend.models.domain import ChunkRecord, RetrievedChunk
from backend.retrieval.prompt_builder import build_rag_messages


def test_rag_prompt_marks_context_as_untrusted() -> None:
    chunk = ChunkRecord(
        id=1,
        document_id=1,
        user_id=1,
        chunk_id="doc-1-chunk-00000",
        text="Revenue increased by 12 percent.",
        token_count=6,
        metadata={"filename": "report.pdf"},
    )
    messages = build_rag_messages("What changed?", [RetrievedChunk(chunk=chunk, score=1.0)])

    assert messages[0]["role"] == "system"
    assert "untrusted" in messages[0]["content"]
    assert "[S1]" in messages[-1]["content"]
    assert "Revenue increased" in messages[-1]["content"]
