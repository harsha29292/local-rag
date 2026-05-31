"""RAG chat orchestration."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from backend.models.domain import User
from backend.retrieval.prompt_builder import build_rag_messages
from backend.schemas.rag import RagRequest
from backend.services.memory_service import MemoryService
from backend.services.ollama_client import OllamaClient
from backend.services.retrieval_service import RetrievalService, source_chunks

logger = logging.getLogger(__name__)


class RagService:
    """Question answering over a user's indexed documents."""

    def __init__(self) -> None:
        self.memory = MemoryService()
        self.retrieval = RetrievalService()
        self.ollama = OllamaClient()

    async def stream_rag_answer(self, user: User, request: RagRequest) -> AsyncIterator[dict]:
        """Retrieve context, stream an answer, and persist messages."""

        conversation = await self.memory.ensure_conversation(user, request.conversation_id, "rag", request.question)
        history = await self.memory.recent_chat_messages(user, conversation.id, limit=8)
        await self.memory.add_message(user, conversation.id, "user", request.question)
        yield {"type": "conversation", "conversation_id": conversation.id, "title": conversation.title}

        retrieved = await self.retrieval.retrieve(user, request.question, request.top_k)
        sources = source_chunks(retrieved)
        yield {"type": "sources", "sources": [source.model_dump() for source in sources]}

        if not retrieved:
            answer = "I could not find relevant indexed document context for that question."
            yield {"type": "token", "content": answer}
            await self.memory.add_message(user, conversation.id, "assistant", answer, {"sources": []})
            yield {"type": "done"}
            return

        messages = build_rag_messages(request.question, retrieved, history=history)
        answer_parts: list[str] = []
        try:
            async for token in self.ollama.stream_chat(messages):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        except Exception as exc:
            logger.exception("RAG streaming failed")
            yield {"type": "error", "message": str(exc)}
            return

        answer = "".join(answer_parts).strip()
        if answer:
            await self.memory.add_message(user, conversation.id, "assistant", answer, {"sources": [source.model_dump() for source in sources]})
        yield {"type": "done"}
