"""Document parsers for supported upload types."""

from __future__ import annotations

import asyncio
from pathlib import Path

from docx import Document as DocxDocument
from fastapi import HTTPException, status
from pypdf import PdfReader


def _parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n[Page {page_number}]\n{text}")
    return "\n".join(pages)


def _parse_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    parts: list[str] = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


async def parse_document(path: Path) -> str:
    """Parse text from a supported document."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return await asyncio.to_thread(_parse_pdf, path)
    if suffix == ".docx":
        return await asyncio.to_thread(_parse_docx, path)
    if suffix == ".txt":
        return await asyncio.to_thread(_parse_txt, path)
    if suffix == ".doc":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Legacy .doc parsing requires external converters. Please upload .docx or PDF.",
        )
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file type: {suffix}")
