"""Path bootstrap for Streamlit page modules."""

from __future__ import annotations

import sys
from pathlib import Path


def add_project_root() -> None:
    """Ensure the project root is importable when Streamlit runs a page file."""

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
