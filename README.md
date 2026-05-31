# Local Multi-User RAG

A local-first Retrieval-Augmented Generation platform using FastAPI, Streamlit, Ollama, FAISS, BM25, SQLite, and optional RAGAS evaluation scaffolding.

## Architecture

```
Streamlit UI
  -> FastAPI orchestration API
    -> Ingestion, retrieval, memory, auth, evaluation services
      -> SQLite + FAISS + BM25
        -> Ollama HTTP API
```

FastAPI is the only layer that talks to Ollama. Streamlit is a client for auth, document management, general chat, and RAG chat.

## Design Decisions

- SQLite is the source of truth for users, documents, chunks, conversations, and messages.
- FAISS and BM25 are per-user derived indexes rebuilt after ingestion, delete, and re-index operations.
- Retrieval uses dense search, sparse BM25 search, Reciprocal Rank Fusion, and an optional MiniLM cross-encoder reranker.
- CPU FAISS and lightweight local models are chosen for constrained hardware such as Quadro P400 5GB VRAM plus Xeon CPU.
- Streaming uses Ollama HTTP streaming through FastAPI `StreamingResponse` to Streamlit.
- Uploaded document text is treated as untrusted context in prompts.

## Prerequisites

1. Python 3.14+
2. Ollama running locally
3. Models pulled:

```powershell
ollama pull phi3:mini
ollama pull nomic-embed-text
```

For reranking, pre-download/cache `cross-encoder/ms-marco-MiniLM-L-6-v2` if the machine will run offline.

## Setup

```powershell
python --version  # should print Python 3.14.x
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If `python` does not point to Python 3.14, invoke your Python 3.14 executable directly when creating the venv.

Edit `.env` and set `JWT_SECRET_KEY` to a strong random value.

Optional reranking and evaluation packages:

```powershell
pip install -r requirements-optional.txt
```

FAISS is included in the core requirements for Python 3.14. The backend also has a NumPy cosine fallback so development still works if FAISS is temporarily unavailable on a machine.

## Run

Start FastAPI:

```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Start Streamlit:

```powershell
streamlit run frontend/Home.py --server.port 8501
```

Open http://localhost:8501.

## Project Structure

```text
backend/
  app.py
  config/
  db/
  ingestion/
  models/
  retrieval/
  routes/
  schemas/
  services/
  utils/
  vectorstore/
  tests/
frontend/
  Home.py
  api/
  components/
  pages/
  utils/
```

## Evaluation

RAGAS scaffolding is available through `backend/services/evaluation_service.py` and the `/evaluation/ragas` route. It accepts question, answer, contexts, and optional ground truth.

## Security Notes

- Do not use wildcard CORS in production.
- Replace the example JWT secret.
- Uploads are extension and size validated, sanitized, and stored outside importable source folders.
- All document retrieval is scoped by authenticated `user_id`.
