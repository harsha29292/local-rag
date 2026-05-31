"""General LLM chat orchestration."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from backend.models.domain import User
from backend.retrieval.prompt_builder import GENERAL_SYSTEM_PROMPT
from backend.schemas.chat import ChatRequest
from backend.services.memory_service import MemoryService
from backend.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ChatService:
    """Persistent general chat service."""

    def __init__(self) -> None:
        self.memory = MemoryService()
        self.ollama = OllamaClient()

    async def stream_general_chat(self, user: User, request: ChatRequest) -> AsyncIterator[dict]:
        """Persist user input, stream assistant tokens, and persist final output."""

        conversation = await self.memory.ensure_conversation(user, request.conversation_id, "general", request.message)
        await self.memory.add_message(user, conversation.id, "user", request.message)
        yield {"type": "conversation", "conversation_id": conversation.id, "title": conversation.title}

        history = await self.memory.recent_chat_messages(user, conversation.id, limit=12)
        messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}, *history]

        answer_parts: list[str] = []
        try:
            async for token in self.ollama.stream_chat(messages):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        except Exception as exc:
            logger.exception("General chat streaming failed")
            yield {"type": "error", "message": str(exc)}
            return

        answer = "".join(answer_parts).strip()
        if answer:
            await self.memory.add_message(user, conversation.id, "assistant", answer)
        yield {"type": "done"}
