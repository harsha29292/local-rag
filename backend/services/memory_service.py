"""Conversation and persistent memory service."""

from __future__ import annotations

import json

from fastapi import HTTPException, status

from backend.db.database import fetch_all, fetch_one, get_db
from backend.models.domain import User
from backend.schemas.chat import ConversationResponse, MessageResponse


class MemoryService:
    """SQLite-backed conversation memory."""

    async def list_conversations(self, user: User, mode: str | None = None) -> list[ConversationResponse]:
        """List user conversations."""

        if mode:
            rows = await fetch_all(
                """
                SELECT id, title, mode, created_at, updated_at
                FROM conversations
                WHERE user_id = ? AND mode = ?
                ORDER BY updated_at DESC
                """,
                (user.id, mode),
            )
        else:
            rows = await fetch_all(
                """
                SELECT id, title, mode, created_at, updated_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user.id,),
            )
        return [_conversation_response(row) for row in rows]

    async def create_conversation(self, user: User, title: str, mode: str) -> ConversationResponse:
        """Create a conversation."""

        db = await get_db()
        cursor = await db.execute(
            "INSERT INTO conversations (user_id, title, mode) VALUES (?, ?, ?)",
            (user.id, title[:120], mode),
        )
        await db.commit()
        row = await self.get_conversation_row(user.id, int(cursor.lastrowid), mode)
        return _conversation_response(row)

    async def ensure_conversation(self, user: User, conversation_id: int | None, mode: str, title_seed: str) -> ConversationResponse:
        """Validate an existing conversation or create a new one."""

        if conversation_id is not None:
            row = await self.get_conversation_row(user.id, conversation_id, mode)
            return _conversation_response(row)
        title = title_seed.strip().replace("\n", " ")[:80] or "New conversation"
        return await self.create_conversation(user, title, mode)

    async def get_conversation_row(self, user_id: int, conversation_id: int, mode: str | None = None):
        """Return a conversation row after ownership validation."""

        if mode:
            row = await fetch_one(
                "SELECT id, title, mode, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ? AND mode = ?",
                (conversation_id, user_id, mode),
            )
        else:
            row = await fetch_one(
                "SELECT id, title, mode, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        return row

    async def list_messages(self, user: User, conversation_id: int) -> list[MessageResponse]:
        """List messages in a conversation."""

        await self.get_conversation_row(user.id, conversation_id)
        rows = await fetch_all(
            """
            SELECT id, conversation_id, role, content, metadata_json, created_at
            FROM messages
            WHERE conversation_id = ? AND user_id = ?
            ORDER BY id ASC
            """,
            (conversation_id, user.id),
        )
        return [_message_response(row, user.id) for row in rows]

    async def recent_chat_messages(self, user: User, conversation_id: int, limit: int = 10) -> list[dict[str, str]]:
        """Return recent messages as Ollama chat messages."""

        await self.get_conversation_row(user.id, conversation_id)
        rows = await fetch_all(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ? AND user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, user.id, limit),
        )
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in reversed(rows)]

    async def add_message(
        self,
        user: User,
        conversation_id: int,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> MessageResponse:
        """Persist a message and bump conversation activity."""

        await self.get_conversation_row(user.id, conversation_id)
        db = await get_db()
        cursor = await db.execute(
            """
            INSERT INTO messages (conversation_id, user_id, role, content, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, user.id, role, content, json.dumps(metadata or {})),
        )
        await db.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?", (conversation_id, user.id))
        await db.commit()
        row = await fetch_one(
            """
            SELECT id, conversation_id, role, content, metadata_json, created_at
            FROM messages
            WHERE id = ?
            """,
            (int(cursor.lastrowid),),
        )
        return _message_response(row, user.id)


def _conversation_response(row) -> ConversationResponse:
    return ConversationResponse(
        id=int(row["id"]),
        title=str(row["title"]),
        mode=str(row["mode"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _message_response(row, _user_id: int) -> MessageResponse:
    return MessageResponse(
        id=int(row["id"]),
        conversation_id=int(row["conversation_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        metadata=json.loads(row["metadata_json"]),
        created_at=str(row["created_at"]),
    )
