"""File validation and safe storage helpers."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import HTTPException, UploadFile, status

from backend.config.settings import get_settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename."""

    name = Path(filename).name.strip() or "document"
    sanitized = _SAFE_NAME_RE.sub("_", name)
    return sanitized[:180]


def validate_extension(filename: str) -> str:
    """Validate an uploaded filename and return its lowercase suffix."""

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    return suffix


async def save_upload_to_temp(upload: UploadFile) -> tuple[Path, str, int]:
    """Stream an upload to a temporary file while enforcing size limits."""

    settings = get_settings()
    validate_extension(upload.filename or "")
    hasher = hashlib.sha256()
    total = 0

    suffix = Path(upload.filename or "upload").suffix.lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > settings.max_upload_bytes:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Upload too large")
            hasher.update(chunk)
            tmp.write(chunk)

    return tmp_path, hasher.hexdigest(), total


def move_temp_upload(tmp_path: Path, user_id: int, document_id: int, filename: str) -> Path:
    """Move a validated temporary file to durable upload storage."""

    settings = get_settings()
    target_dir = settings.upload_dir / f"user_{user_id}" / str(document_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / sanitize_filename(filename)
    shutil.move(str(tmp_path), target_path)
    return target_path
