"""Utility tests."""

from __future__ import annotations

from backend.ingestion.cleaner import clean_text
from backend.utils.files import sanitize_filename


def test_sanitize_filename_removes_path_segments() -> None:
    assert sanitize_filename("../bad name.pdf") == "bad_name.pdf"


def test_clean_text_collapses_spacing() -> None:
    assert clean_text("a   b\n\n\nc") == "a b\n\nc"
