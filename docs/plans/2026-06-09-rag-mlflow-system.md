# RAG Document Q&A with MLflow Tracking — Project Summary

> **Status:** COMPLETE and verified live (2026-06-10)
> **URL:** http://localhost:8001/docs (FastAPI), http://localhost:5001 (MLflow)

## What Was Built

A production-grade RAG pipeline running entirely on an M1 Mac (16GB RAM) with zero cloud cost:

1. **Document ingestion** — Upload PDFs or paste text, chunked with RecursiveCharacterTextSplitter
2. **Local embeddings** — all-MiniLM-L6-v2 via sentence-transformers (bypasses slow Ollama /api/embeddings)
3. **Vector storage** — ChromaDB, local-first, zero config
4. **Streaming Q&A** — FastAPI SSE endpoint, retrieves top-3 chunks, feeds to qwen3:4b via Ollama
5. **Evaluation pipeline** — LLM-as-judge scoring (faithfulness + relevance), keyword-based retrieval accuracy
6. **Experiment tracking** — MLflow logs params (chunk size, overlap, model names) and metrics per eval run
7. **Chat UI** — Dark-themed vanilla HTML/CSS/JS frontend with PDF upload, streaming chat, and eval controls
8. **Test suite** — 6 tests, all mocked, runs in ~30s with no external dependencies

## Verified Working

- PDF upload and indexing (3 PDFs tested)
- Streaming Q&A with source citations
- Evaluation suite with MLflow logging
- All 6 tests passing
- Frontend chat UI connecting to backend

## Port Assignments

| Service | Port | Reason |
|---------|------|--------|
| FastAPI | 8001 | Honcho occupies 8000 |
| MLflow | 5001 | macOS Control Center occupies 5000 |

## Key Deviations from Original Plan

| Original Plan | What We Actually Did | Why |
|---------------|---------------------|-----|
| Ollama `/api/embeddings` for vectors | sentence-transformers (all-MiniLM-L6-v2) | Ollama embeddings endpoint too slow on M1 |
| nomic-embed-text model | all-MiniLM-L6-v2 | 80MB vs 274MB, much faster on CPU |
| PostgreSQL + pgvector in docker-compose | Removed — not used | App uses ChromaDB + SQLite, no Postgres needed |
| Tests call real Ollama | All external calls mocked | Tests run in 30s vs minutes, no model files required |
| Port 8000 (FastAPI), 5000 (MLflow) | 8001 and 5001 | Port conflicts on macOS |

## Git History

```
14af25b  fix: change ports to avoid conflicts (MLflow 5000->5001, FastAPI 8000->8001)
a0d70d7  fix: correct eval params, mock slow tests, clean docker-compose
329fbd5  fix: make imports work when running from backend/ or repo root
c453b9a  docs: improve README with badges, architecture diagram, API docs
faab69c  fix: switch embeddings to sentence-transformers (bypass Ollama /api/embeddings)
3680869  feat: add seed script + switch to all-minilm for M1 Mac
c1402b1  feat: complete RAG + MLflow system implementation
67d2ce3  initial commit: project 1 scaffolding
```

## Next Steps (Portfolio)

1. Write blog post (Medium/Dev.to) — "I Built a Production RAG System on My M1 Mac for $0"
2. Record demo video / GIF of the chat UI + eval suite
3. Add to saintlex.sbs portfolio page
4. Move to Project 2: Hermes MLOps case study
