"""FastAPI application factory and runtime hooks."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import get_settings
from backend.db.database import close_db, init_db
from backend.routes import auth, chat, documents, evaluation, health, rag
from backend.utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize durable resources before serving requests."""

    settings = get_settings()
    configure_logging(settings.log_level)
    logging.getLogger(__name__).info("Starting local RAG backend")
    settings.ensure_directories()
    await init_db()
    yield
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""

    settings = get_settings()
    app = FastAPI(
        title="Local Multi-User RAG",
        version="0.1.0",
        description="Local-first RAG with FastAPI, Ollama, FAISS, BM25, and Streamlit.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(health.router)
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(rag.router, prefix="/rag", tags=["rag"])
    app.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
    return app


app = create_app()
