# RAG Document Q&A System with Eval Pipeline

> Production-grade RAG system with built-in evaluation framework — the #1 signal hiring managers look for in AI Engineers.

## What This Is

A complete **Retrieval-Augmented Generation (RAG) system** with a built-in **evaluation pipeline** that measures retrieval accuracy, answer quality, and catches failure modes — the single biggest signal that separates "tutorial followers" from "production AI engineers" in 2026.

**Key Differentiator:** Most portfolio projects stop at "it answers questions." This one proves the answers are *correct*, *relevant*, and *measurable*.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  PDF Upload  │────▶│  Chunking    │────▶│  Embedding  │
│  (frontend)  │     │  (recursive) │     │  (nomic)    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer     │◀────│  LLM Gen     │◀────│  Retrieval  │
│  (streaming) │     │  (Ollama)    │     │  (pgvector) │
└──────┬──────┘     └──────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Eval Judge  │────▶│  Golden Set  │────▶│  Metrics    │
│  (LLM-as-    │     │  Comparison  │     │  Dashboard  │
│   judge)     │     │              │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Quick Start

```bash
git clone https://github.com/SaintChris/rag-eval-system.git
cd rag-eval-system

# Start infrastructure
docker-compose up -d

# Install backend deps
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run backend
uvicorn app:app --reload --port 8000

# Open frontend
open http://localhost:8000
```

## Features

- **PDF Upload & Processing** — Upload documents, automatic chunking with overlap
- **Semantic Search** — pgvector with nomic-embed for meaning-based retrieval
- **Streaming Answers** — Real-time token streaming from local LLM
- **Eval Pipeline** — Golden dataset evaluation with LLM-as-judge scoring
- **Failure Mode Analysis** — Identifies when retrieval fails vs when generation fails
- **Metrics Dashboard** — Retrieval accuracy, answer relevance, latency tracking
- **Zero Cloud Cost** — Runs entirely on local Ollama + Docker

## Tech Stack

`Python` · `FastAPI` · `pgvector` · `Ollama` · `nomic-embed` · `Docker` · `HTML/CSS/JS`

## Eval Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Retrieval Accuracy | Does the right chunk get retrieved? | >85% |
| Answer Relevance | Is the answer relevant to the question? | >90% |
| Faithfulness | Is the answer grounded in retrieved context? | >80% |
| Latency | End-to-end response time | <3s |

## Why This Gets You Hired

According to 2026 hiring data, **eval design is the single best signal of real LLM experience**. This project demonstrates:

1. **You can build RAG** — not just call an API
2. **You can measure quality** — not just ship and hope
3. **You understand failure modes** — retrieval vs generation failures
4. **You think in production** — metrics, monitoring, observability

## License

MIT
