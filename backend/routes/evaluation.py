"""Evaluation routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.models.domain import User
from backend.schemas.evaluation import RagasEvaluationRequest, RagasEvaluationResponse
from backend.services.auth_service import get_current_user
from backend.services.evaluation_service import EvaluationService

router = APIRouter()


@router.post("/ragas", response_model=RagasEvaluationResponse)
async def evaluate_ragas(
    payload: RagasEvaluationRequest,
    _current_user: User = Depends(get_current_user),
) -> RagasEvaluationResponse:
    """Run RAGAS evaluation scaffolding for a single sample."""

    return await EvaluationService().evaluate_ragas(payload)
