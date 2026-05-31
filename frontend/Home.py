"""Streamlit entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.components.auth import render_auth_sidebar

st.set_page_config(page_title="Local RAG", page_icon=":material/database:", layout="wide")

st.title("Local RAG")
render_auth_sidebar()

st.page_link("pages/1_RAG_Workspace.py", label="RAG Workspace", icon=":material/folder_open:")
st.page_link("pages/2_General_Chat.py", label="General Chat", icon=":material/chat:")

st.caption("FastAPI · Ollama · FAISS · BM25 · SQLite")
