"""RAGAS evaluation scaffolding."""

from __future__ import annotations

import asyncio
from typing import Any

from backend.schemas.evaluation import RagasEvaluationRequest, RagasEvaluationResponse


class EvaluationService:
    """Run RAGAS metrics when dependencies and evaluator configuration are available."""

    async def evaluate_ragas(self, request: RagasEvaluationRequest) -> RagasEvaluationResponse:
        """Evaluate a single RAG sample.

        RAGAS usually needs an evaluator LLM and embeddings. This scaffold keeps the
        integration isolated so local evaluator adapters can be injected later.
        """

        try:
            return await asyncio.to_thread(self._evaluate_sync, request)
        except Exception as exc:
            return RagasEvaluationResponse(
                scores={
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None,
                },
                notes=f"RAGAS evaluation is scaffolded but not fully configured: {exc}",
            )

    def _evaluate_sync(self, request: RagasEvaluationRequest) -> RagasEvaluationResponse:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        data: dict[str, list[Any]] = {
            "question": [request.question],
            "answer": [request.answer],
            "contexts": [request.contexts],
            "ground_truth": [request.ground_truth or ""],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        scores: dict[str, float | None] = {}
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            value = result.get(name)
            if isinstance(value, list) and value:
                scores[name] = float(value[0])
            elif value is not None:
                scores[name] = float(value)
            else:
                scores[name] = None
        return RagasEvaluationResponse(scores=scores)
