"""Prompt assembly for RAG answers."""

from __future__ import annotations

from backend.config.settings import get_settings
from backend.models.domain import RetrievedChunk

RAG_SYSTEM_PROMPT = """You are a precise, factual local assistant. Answer the user's question using ONLY the provided context chunks.

Rules:
1. Use ONLY facts directly mentioned in the context. Do NOT use outside knowledge, assumptions, or extrapolate.
2. Do not extrapolate tool characteristics (e.g. if the context mentions a tool like Apache Flink, do not assume or state that the project itself is written in Java unless explicitly stated).
3. If the context does not contain the answer to the question, reply EXACTLY with: "The uploaded documents do not contain information about this topic."
4. Do not assume or guess. If it is not explicitly written in the context, treat it as not found.
5. If a detail is described generally in one place and specifically in another (e.g. in a table or list), include the specific details and numbers.
6. Cite the source chunk labels (e.g., [S1], [S2]) for all facts you state.
7. Keep your answer brief and factual (under 300 words)."""

GENERAL_SYSTEM_PROMPT = """You are a helpful local assistant running through a private Ollama backend. Be concise, practical, and transparent when unsure."""


def build_rag_messages(question: str, chunks: list[RetrievedChunk], history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Build Ollama chat messages for a RAG query."""

    settings = get_settings()
    context_blocks: list[str] = []
    used_chars = 0
    for idx, item in enumerate(chunks, start=1):
        filename = item.chunk.metadata.get("filename", f"document-{item.chunk.document_id}")
        remaining_chars = settings.rag_context_max_chars - used_chars
        if remaining_chars <= 0:
            break
        chunk_text = item.chunk.text[:remaining_chars]
        used_chars += len(chunk_text)
        context_blocks.append(f"[S{idx}] (Source: {filename})\n{chunk_text}")

    context = "\n\n".join(context_blocks)
    user_prompt = f"""Answer the QUESTION using only the CONTEXT CHUNKS provided below. Be factual and brief. Cite source labels [S1], [S2], etc.
If the answer is not in the context, say: "The uploaded documents do not contain information about this topic."

STRICT RULES:
1. When stating any performance ratings, levels, or classes, always include both the number/level and its descriptive name in parentheses as written in the text (for example, write "Rating 5 (Exceptional)" or "Tier 1 (Confidential)").
2. Keep your answer brief and factual (under 250 words).

CONTEXT CHUNKS:
{context}

QUESTION:
{question}

Answer:"""

    messages = []
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_prompt})
    return messages
