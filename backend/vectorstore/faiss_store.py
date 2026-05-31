"""Per-user FAISS vector store."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from backend.config.settings import get_settings
from backend.models.domain import ChunkRecord, RetrievedChunk


class FaissVectorStore:
    """Persistent per-user FAISS store using cosine search via inner product."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _user_dir(self, user_id: int) -> Path:
        return self.settings.vectorstore_dir / f"user_{user_id}"

    def _index_path(self, user_id: int) -> Path:
        return self._user_dir(user_id) / "index.faiss"

    def _docstore_path(self, user_id: int) -> Path:
        return self._user_dir(user_id) / "docstore.json"

    async def build_user_index(self, user_id: int, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Rebuild a user's FAISS index from chunk rows."""

        await asyncio.to_thread(self._build_user_index_sync, user_id, chunks, embeddings)

    def _build_user_index_sync(self, user_id: int, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_path(user_id)
        docstore_path = self._docstore_path(user_id)

        if not chunks:
            index_path.unlink(missing_ok=True)
            docstore_path.unlink(missing_ok=True)
            return

        matrix = np.asarray(embeddings, dtype="float32")
        _normalize_rows(matrix)
        faiss = _try_load_faiss()
        if faiss is not None:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            faiss.write_index(index, str(index_path))
        else:
            np.save(str(index_path) + ".npy", matrix)

        records = [_chunk_to_json(chunk) for chunk in chunks]
        docstore_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    async def search(self, user_id: int, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Search a user's FAISS index."""

        return await asyncio.to_thread(self._search_sync, user_id, query_embedding, top_k)

    def _search_sync(self, user_id: int, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        index_path = self._index_path(user_id)
        docstore_path = self._docstore_path(user_id)
        npy_path = Path(str(index_path) + ".npy")
        if not docstore_path.exists() or (not index_path.exists() and not npy_path.exists()):
            return []

        records = json.loads(docstore_path.read_text(encoding="utf-8"))
        query = np.asarray([query_embedding], dtype="float32")
        _normalize_rows(query)
        faiss = _try_load_faiss()
        if faiss is not None and index_path.exists():
            index = faiss.read_index(str(index_path))
            scores, indices = index.search(query, min(top_k, len(records)))
            score_values = scores[0].tolist()
            index_values = indices[0].tolist()
        else:
            matrix = np.load(npy_path)
            similarities = matrix @ query[0]
            index_values = np.argsort(-similarities)[: min(top_k, len(records))].tolist()
            score_values = [float(similarities[idx]) for idx in index_values]

        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(index_values, start=1):
            if idx < 0:
                continue
            chunk = _chunk_from_json(records[idx])
            results.append(RetrievedChunk(chunk=chunk, score=float(score_values[rank - 1]), source="dense", ranks={"dense": rank}))
        return results


def _chunk_to_json(chunk: ChunkRecord) -> dict:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "user_id": chunk.user_id,
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata,
    }


def _chunk_from_json(data: dict) -> ChunkRecord:
    return ChunkRecord(
        id=int(data["id"]),
        document_id=int(data["document_id"]),
        user_id=int(data["user_id"]),
        chunk_id=str(data["chunk_id"]),
        text=str(data["text"]),
        token_count=int(data["token_count"]),
        metadata=dict(data.get("metadata", {})),
    )


def _try_load_faiss():
    try:
        import faiss

        return faiss
    except ImportError:
        return None


def _normalize_rows(matrix: np.ndarray) -> None:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
