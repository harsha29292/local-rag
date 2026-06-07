"""Synchronous Streamlit client for the FastAPI backend."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

API_BASE_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")


class ApiError(RuntimeError):
    """Backend API error shown in Streamlit."""


def _headers(token: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _handle_response(response: httpx.Response) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise ApiError(str(detail)) from exc
    if not response.content:
        return None
    return response.json()


def register(username: str, password: str, registration_code: str | None = None) -> dict[str, Any]:
    response = httpx.post(
        f"{API_BASE_URL}/auth/register",
        json={"username": username, "password": password, "registration_code": registration_code},
        timeout=30,
    )
    return _handle_response(response)


def login(username: str, password: str) -> dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/auth/login", json={"username": username, "password": password}, timeout=30)
    return _handle_response(response)


def me(token: str) -> dict[str, Any]:
    response = httpx.get(f"{API_BASE_URL}/auth/me", headers=_headers(token), timeout=30)
    return _handle_response(response)


def list_documents(token: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{API_BASE_URL}/documents", headers=_headers(token), timeout=60)
    return _handle_response(response)


def upload_document(token: str, filename: str, data: bytes) -> dict[str, Any]:
    files = {"file": (filename, data)}
    response = httpx.post(f"{API_BASE_URL}/documents", headers=_headers(token), files=files, timeout=600)
    return _handle_response(response)


def upload_documents(token: str, uploads: list[tuple[str, bytes]]) -> dict[str, Any]:
    files = [("files", (filename, data)) for filename, data in uploads]
    response = httpx.post(f"{API_BASE_URL}/documents/batch", headers=_headers(token), files=files, timeout=1200)
    return _handle_response(response)


def delete_document(token: str, document_id: int) -> None:
    response = httpx.delete(f"{API_BASE_URL}/documents/{document_id}", headers=_headers(token), timeout=300)
    _handle_response(response)


def reindex_document(token: str, document_id: int) -> dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/documents/{document_id}/reindex", headers=_headers(token), timeout=600)
    return _handle_response(response)


def list_conversations(token: str, mode: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{API_BASE_URL}/chat/conversations", headers=_headers(token), params={"mode": mode}, timeout=30)
    return _handle_response(response)


def list_messages(token: str, conversation_id: int) -> list[dict[str, Any]]:
    response = httpx.get(f"{API_BASE_URL}/chat/conversations/{conversation_id}/messages", headers=_headers(token), timeout=30)
    return _handle_response(response)


def stream_general_chat(token: str, message: str, conversation_id: int | None) -> Iterator[dict[str, Any]]:
    payload = {"message": message, "conversation_id": conversation_id}
    yield from _stream_ndjson(f"{API_BASE_URL}/chat/general/stream", token, payload)


def stream_rag_query(token: str, question: str, conversation_id: int | None) -> Iterator[dict[str, Any]]:
    payload = {"question": question, "conversation_id": conversation_id}
    yield from _stream_ndjson(f"{API_BASE_URL}/rag/query/stream", token, payload)


def _stream_ndjson(url: str, token: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    headers = _headers(token)
    headers["Accept"] = "application/x-ndjson"
    with httpx.stream("POST", url, headers=headers, json=payload, timeout=None) as response:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except Exception:
                pass
            raise ApiError(str(detail)) from exc
        for line in response.iter_lines():
            if not line:
                continue
            yield json.loads(line)
