"""Document ingestion and lifecycle service."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from backend.config.settings import get_settings
from backend.db.database import fetch_all, fetch_one, get_db
from backend.ingestion.chunker import chunk_text
from backend.ingestion.cleaner import clean_text
from backend.ingestion.parsers import parse_document
from backend.models.domain import User
from backend.schemas.documents import DocumentResponse
from backend.services.index_service import IndexService
from backend.utils.files import move_temp_upload, sanitize_filename, save_upload_to_temp, validate_extension

logger = logging.getLogger(__name__)


class DocumentService:
    """Manage uploaded documents and derived chunks."""

    def __init__(self) -> None:
        self.index_service = IndexService()

    async def list_documents(self, user: User) -> list[DocumentResponse]:
        """List documents owned by a user."""

        rows = await fetch_all(
            """
            SELECT id, filename, original_filename, status, chunk_count, page_count, error_message, created_at, updated_at
            FROM documents
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (user.id,),
        )
        return [_document_response(row) for row in rows]

    async def ingest_upload(self, user: User, upload: UploadFile) -> DocumentResponse:
        """Validate, persist, parse, chunk, and index an uploaded document."""

        original_filename = upload.filename or "document"
        validate_extension(original_filename)
        await self._assert_user_document_capacity(user.id)
        safe_filename = sanitize_filename(original_filename)
        tmp_path, content_hash, _size = await save_upload_to_temp(upload)

        db = await get_db()
        cursor = await db.execute(
            """
            INSERT INTO documents (user_id, filename, original_filename, file_path, content_hash, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user.id, safe_filename, original_filename, "", content_hash, "processing"),
        )
        document_id = int(cursor.lastrowid)
        await db.commit()

        stored_path = move_temp_upload(tmp_path, user.id, document_id, safe_filename)
        await db.execute("UPDATE documents SET file_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (str(stored_path), document_id))
        await db.commit()

        try:
            await self._ingest_existing_file(user.id, document_id, stored_path, safe_filename)
        except Exception as exc:
            logger.exception("Document ingestion failed for document %s", document_id)
            await db.execute(
                """
                UPDATE documents
                SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                ("failed", str(exc)[:1000], document_id, user.id),
            )
            await db.commit()
            raise

        row = await self._get_document_row(user.id, document_id)
        return _document_response(row)

    async def delete_document(self, user: User, document_id: int) -> None:
        """Delete a document and rebuild user indexes."""

        row = await self._get_document_row(user.id, document_id)
        db = await get_db()
        await db.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (document_id, user.id))
        await db.commit()
        self._delete_upload_dir(row["file_path"])
        await self.index_service.rebuild_user_indexes(user.id)

    async def reindex_document(self, user: User, document_id: int) -> DocumentResponse:
        """Re-parse a stored document and rebuild indexes."""

        row = await self._get_document_row(user.id, document_id)
        path = Path(row["file_path"])
        if not path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stored file missing")
        await self._ingest_existing_file(user.id, document_id, path, str(row["filename"]))
        updated = await self._get_document_row(user.id, document_id)
        return _document_response(updated)

    async def ingest_uploads(self, user: User, uploads: list[UploadFile]) -> list[DocumentResponse]:
        """Ingest a small batch of uploaded files sequentially for predictable memory use."""

        settings = get_settings()
        if len(uploads) > settings.max_files_per_upload:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload at most {settings.max_files_per_upload} files at a time.",
            )
        row = await fetch_one("SELECT COUNT(*) AS count FROM documents WHERE user_id = ? AND status != 'failed'", (user.id,))
        current_count = int(row["count"]) if row is not None else 0
        if current_count + len(uploads) > settings.max_documents_per_user:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Document limit reached. Keep at most {settings.max_documents_per_user} documents indexed.",
            )
        documents: list[DocumentResponse] = []
        for upload in uploads:
            documents.append(await self.ingest_upload(user, upload))
        return documents

    async def _ingest_existing_file(self, user_id: int, document_id: int, path: Path, filename: str) -> None:
        settings = get_settings()
        db = await get_db()
        await db.execute(
            """
            UPDATE documents
            SET status = ?, error_message = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            ("processing", document_id, user_id),
        )
        await db.execute("DELETE FROM chunks WHERE document_id = ? AND user_id = ?", (document_id, user_id))
        await db.commit()

        parsed = await parse_document(path, settings.max_pages_per_document)
        await self._assert_user_page_capacity(user_id, parsed.page_count, document_id)
        cleaned = clean_text(parsed.text)
        if not cleaned:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No extractable text found")

        chunks = chunk_text(cleaned)
        now = datetime.now(timezone.utc).isoformat()
        for idx, chunk in enumerate(chunks):
            chunk_id = f"doc-{document_id}-chunk-{idx:05d}"
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "user_id": user_id,
                "chunk_id": chunk_id,
                "created_at": now,
            }
            await db.execute(
                """
                INSERT INTO chunks (document_id, user_id, chunk_id, text, token_count, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, user_id, chunk_id, chunk.text, chunk.token_count, json.dumps(metadata)),
            )

        await db.execute(
            """
            UPDATE documents
            SET status = ?, chunk_count = ?, error_message = NULL, updated_at = CURRENT_TIMESTAMP
            , page_count = ?
            WHERE id = ? AND user_id = ?
            """,
            ("ready", len(chunks), parsed.page_count, document_id, user_id),
        )
        await db.commit()
        await self.index_service.rebuild_user_indexes(user_id)

    async def _get_document_row(self, user_id: int, document_id: int):
        row = await fetch_one(
            """
            SELECT id, user_id, filename, original_filename, file_path, status, chunk_count, page_count, error_message, created_at, updated_at
            FROM documents
            WHERE id = ? AND user_id = ?
            """,
            (document_id, user_id),
        )
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        return row

    async def _assert_user_document_capacity(self, user_id: int) -> None:
        settings = get_settings()
        row = await fetch_one("SELECT COUNT(*) AS count FROM documents WHERE user_id = ? AND status != 'failed'", (user_id,))
        current_count = int(row["count"]) if row is not None else 0
        if current_count >= settings.max_documents_per_user:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Document limit reached. Keep at most {settings.max_documents_per_user} documents indexed.",
            )

    async def _assert_user_page_capacity(self, user_id: int, new_page_count: int, document_id: int) -> None:
        settings = get_settings()
        row = await fetch_one(
            "SELECT COALESCE(SUM(page_count), 0) AS page_count FROM documents WHERE user_id = ? AND id != ?",
            (user_id, document_id),
        )
        current_pages = int(row["page_count"]) if row is not None else 0
        if current_pages + new_page_count > settings.max_pages_per_user:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Page limit exceeded. Keep indexed documents under {settings.max_pages_per_user} total pages.",
            )

    def _delete_upload_dir(self, file_path: str) -> None:
        settings = get_settings()
        if not file_path:
            return
        path = Path(file_path).resolve()
        upload_root = settings.upload_dir.resolve()
        try:
            path.relative_to(upload_root)
        except ValueError:
            logger.warning("Refusing to delete file outside upload root: %s", path)
            return
        document_dir = path.parent
        if document_dir.exists():
            shutil.rmtree(document_dir)


def _document_response(row) -> DocumentResponse:
    return DocumentResponse(
        id=int(row["id"]),
        filename=str(row["filename"]),
        original_filename=str(row["original_filename"]),
        status=str(row["status"]),
        chunk_count=int(row["chunk_count"]),
        page_count=int(row["page_count"]),
        error_message=row["error_message"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
