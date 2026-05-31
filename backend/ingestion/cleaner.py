"""Text cleaning utilities."""

from __future__ import annotations

import re

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize extracted document text while preserving paragraph boundaries."""

    text = _strip_control_chars(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
