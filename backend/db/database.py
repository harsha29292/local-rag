"""SQLite connection management and schema initialization."""

from __future__ import annotations

import aiosqlite

from backend.config.settings import get_settings

_connection: aiosqlite.Connection | None = None


def _sqlite_path_from_url(database_url: str) -> str:
    """Extract a filesystem path from a sqlite+aiosqlite URL."""

    prefix = "sqlite+aiosqlite:///"
    if not database_url.startswith(prefix):
        msg = "Only sqlite+aiosqlite URLs are supported by this MVP"
        raise ValueError(msg)
    return database_url.removeprefix(prefix)


async def get_db() -> aiosqlite.Connection:
    """Return a shared SQLite connection."""

    global _connection
    if _connection is None:
        settings = get_settings()
        settings.ensure_directories()
        db_path = _sqlite_path_from_url(settings.database_url)
        _connection = await aiosqlite.connect(db_path)
        _connection.row_factory = aiosqlite.Row
        await _connection.execute("PRAGMA journal_mode=WAL")
        await _connection.execute("PRAGMA foreign_keys=ON")
        await _connection.execute("PRAGMA busy_timeout=5000")
        await _connection.commit()
    return _connection


async def fetch_one(sql: str, parameters: tuple = ()) -> aiosqlite.Row | None:
    """Fetch a single row."""

    db = await get_db()
    cursor = await db.execute(sql, parameters)
    return await cursor.fetchone()


async def fetch_all(sql: str, parameters: tuple = ()) -> list[aiosqlite.Row]:
    """Fetch all rows for a query."""

    db = await get_db()
    cursor = await db.execute(sql, parameters)
    rows = await cursor.fetchall()
    return list(rows)


async def close_db() -> None:
    """Close the SQLite connection."""

    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None


async def init_db() -> None:
    """Initialize database schema."""

    db = await get_db()
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            page_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON chunks(user_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            mode TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_user_mode ON conversations(user_id, mode);

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
        """
    )
    await _ensure_column(db, "documents", "page_count", "INTEGER NOT NULL DEFAULT 0")
    await db.commit()


async def _ensure_column(db: aiosqlite.Connection, table: str, column: str, definition: str) -> None:
    """Add a column to an existing SQLite table when upgrading old local DBs."""

    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    existing = {str(row["name"]) for row in rows}
    if column not in existing:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
