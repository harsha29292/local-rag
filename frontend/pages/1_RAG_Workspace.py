"""RAG workspace page."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.api import client
from frontend.components.auth import require_auth
from frontend.components.conversations import render_history, select_conversation

st.set_page_config(page_title="RAG Workspace", page_icon=":material/folder_open:", layout="wide")

token = require_auth()
st.sidebar.divider()
conversation_id = select_conversation(token, "rag", "rag_conversation_id")

st.title("RAG Workspace")

left, right = st.columns([0.38, 0.62], gap="large")

with left:
    st.subheader("Documents")
    uploaded = st.file_uploader("Upload", type=["pdf", "docx", "doc", "txt"], accept_multiple_files=False)
    if uploaded and st.button("Index document", type="primary", use_container_width=True):
        try:
            with st.spinner("Indexing document"):
                client.upload_document(token, uploaded.name, uploaded.getvalue())
            st.success("Indexed")
            st.rerun()
        except client.ApiError as exc:
            st.error(str(exc))

    try:
        documents = client.list_documents(token)
    except client.ApiError as exc:
        st.error(str(exc))
        documents = []

    for doc in documents:
        with st.container(border=True):
            st.markdown(f"**{doc['filename']}**")
            st.caption(f"{doc['status']} · {doc['chunk_count']} chunks")
            if doc.get("error_message"):
                st.error(doc["error_message"])
            col_a, col_b = st.columns(2)
            if col_a.button("Re-index", key=f"reindex_{doc['id']}", use_container_width=True):
                try:
                    with st.spinner("Re-indexing"):
                        client.reindex_document(token, int(doc["id"]))
                    st.rerun()
                except client.ApiError as exc:
                    st.error(str(exc))
            if col_b.button("Delete", key=f"delete_{doc['id']}", use_container_width=True):
                try:
                    client.delete_document(token, int(doc["id"]))
                    st.rerun()
                except client.ApiError as exc:
                    st.error(str(exc))

with right:
    st.subheader("Ask")
    render_history(token, conversation_id)
    question = st.chat_input("Ask over your documents")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        sources_box = st.empty()
        with st.chat_message("assistant"):
            answer_box = st.empty()
            answer = ""
            sources = []
            try:
                for event in client.stream_rag_query(token, question, conversation_id):
                    if event["type"] == "conversation":
                        st.session_state.rag_conversation_id = int(event["conversation_id"])
                    elif event["type"] == "sources":
                        sources = event["sources"]
                        with sources_box.container():
                            if sources:
                                with st.expander("Sources", expanded=False):
                                    for idx, source in enumerate(sources, start=1):
                                        st.markdown(f"**S{idx}. {source['filename']}**")
                                        st.caption(f"{source['chunk_id']} · score {source['score']:.4f}")
                                        st.write(source["text"][:1200])
                    elif event["type"] == "token":
                        answer += event["content"]
                        answer_box.markdown(answer)
                    elif event["type"] == "error":
                        st.error(event["message"])
            except client.ApiError as exc:
                st.error(str(exc))
