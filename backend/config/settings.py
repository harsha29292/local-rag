"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for backend services."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    vectorstore_dir: Path = Path("./data/vectorstores")
    bm25_dir: Path = Path("./data/bm25")

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"])

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 720

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:latest"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_seconds: float = 120.0
    ollama_request_retries: int = 2

    max_upload_mb: int = 50
    max_files_per_upload: int = 5
    max_documents_per_user: int = 5
    max_pages_per_user: int = 800
    max_pages_per_document: int = 800
    allowed_upload_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".docx", ".txt"])
    chunk_size_tokens: int = 650
    chunk_overlap_tokens: int = 90

    retrieval_dense_top_k: int = 24
    retrieval_sparse_top_k: int = 24
    retrieval_final_top_k: int = 6
    retrieval_min_score: float = 0.012
    retrieval_min_query_overlap: int = 1
    rag_context_max_chars: int = 12000
    rrf_k: int = 60

    public_registration_enabled: bool = True
    registration_invite_code: str | None = None

    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_max_candidates: int = 12

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Allow comma-delimited CORS origins in .env files."""

        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("allowed_upload_extensions", mode="before")
    @classmethod
    def parse_allowed_upload_extensions(cls, value: str | list[str]) -> list[str]:
        """Allow comma-delimited file extensions in .env files."""

        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return [item.lower() for item in value]

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""

        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create runtime data directories if they do not exist."""

        for path in (self.data_dir, self.upload_dir, self.vectorstore_dir, self.bm25_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings()
