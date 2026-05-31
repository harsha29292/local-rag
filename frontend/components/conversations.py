"""Conversation selection widgets."""

from __future__ import annotations

import streamlit as st

from frontend.api import client


def select_conversation(token: str, mode: str, state_key: str) -> int | None:
    """Render a conversation selector and return the selected id."""

    conversations = client.list_conversations(token, mode)
    labels = ["New conversation"] + [f"{item['title']} · #{item['id']}" for item in conversations]
    ids = [None] + [int(item["id"]) for item in conversations]

    current_id = st.session_state.get(state_key)
    index = ids.index(current_id) if current_id in ids else 0
    selected = st.sidebar.selectbox("Conversation", labels, index=index, key=f"{state_key}_select")
    selected_id = ids[labels.index(selected)]
    st.session_state[state_key] = selected_id
    return selected_id


def render_history(token: str, conversation_id: int | None) -> None:
    """Render previous chat messages."""

    if conversation_id is None:
        return
    for message in client.list_messages(token, conversation_id):
        if message["role"] in {"user", "assistant"}:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
