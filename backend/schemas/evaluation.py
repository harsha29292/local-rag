"""RAGAS evaluation schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RagasEvaluationRequest(BaseModel):
    """Minimal payload for RAGAS single-sample scoring."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[str] = Field(min_length=1)
    ground_truth: str | None = None


class RagasEvaluationResponse(BaseModel):
    """RAGAS score response."""

    scores: dict[str, float | None]
    notes: str | None = None
