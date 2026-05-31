"""General chat page."""

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

st.set_page_config(page_title="General Chat", page_icon=":material/chat:", layout="wide")

token = require_auth()
st.sidebar.divider()
conversation_id = select_conversation(token, "general", "general_conversation_id")

st.title("General Chat")

render_history(token, conversation_id)

message = st.chat_input("Message")
if message:
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"):
        answer_box = st.empty()
        answer = ""
        try:
            for event in client.stream_general_chat(token, message, conversation_id):
                if event["type"] == "conversation":
                    st.session_state.general_conversation_id = int(event["conversation_id"])
                elif event["type"] == "token":
                    answer += event["content"]
                    answer_box.markdown(answer)
                elif event["type"] == "error":
                    st.error(event["message"])
        except client.ApiError as exc:
            st.error(str(exc))
