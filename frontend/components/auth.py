"""Authentication widgets for Streamlit pages."""

from __future__ import annotations

import streamlit as st

from frontend.api import client


def init_session() -> None:
    """Initialize Streamlit auth state."""

    st.session_state.setdefault("token", None)
    st.session_state.setdefault("username", None)


def render_auth_sidebar() -> str | None:
    """Render login/register controls and return the active token."""

    init_session()
    with st.sidebar:
        st.subheader("Account")
        if st.session_state.token:
            st.caption(st.session_state.username)
            if st.button("Sign out", use_container_width=True):
                st.session_state.token = None
                st.session_state.username = None
                st.rerun()
            return st.session_state.token

        tab_login, tab_register = st.tabs(["Login", "Register"])
        with tab_login:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", use_container_width=True):
                _authenticate(client.login, username, password)
        with tab_register:
            username = st.text_input("Username", key="register_username")
            password = st.text_input("Password", type="password", key="register_password")
            registration_code = st.text_input("Registration code", type="password", key="register_code")
            if st.button("Create account", use_container_width=True):
                _register(username, password, registration_code)
    return st.session_state.token


def require_auth() -> str:
    """Stop page rendering until the user is authenticated."""

    token = render_auth_sidebar()
    if not token:
        st.info("Login or create an account to continue.")
        st.stop()
    return token


def _authenticate(fn, username: str, password: str) -> None:
    if not username or not password:
        st.warning("Username and password are required.")
        return
    try:
        result = fn(username, password)
    except client.ApiError as exc:
        st.error(str(exc))
        return
    st.session_state.token = result["access_token"]
    st.session_state.username = result["username"]
    st.rerun()


def _register(username: str, password: str, registration_code: str) -> None:
    if not username or not password:
        st.warning("Username and password are required.")
        return
    try:
        result = client.register(username, password, registration_code or None)
    except client.ApiError as exc:
        st.error(str(exc))
        return
    st.session_state.token = result["access_token"]
    st.session_state.username = result["username"]
    st.rerun()
