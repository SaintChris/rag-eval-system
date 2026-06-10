# RAG Document Q&A with MLflow Tracking

> **Portfolio Project #1** — A production-grade Retrieval-Augmented Generation system with built-in evaluation pipeline and experiment tracking. Runs entirely on a laptop. Zero cloud cost.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0+-green.svg)](https://www.langchain.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.16+-blue.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What This Is

Most RAG tutorials stop at "it answers questions." This one proves the answers are **correct**, **relevant**, and **measurable** — the single biggest signal that separates tutorial followers from production AI engineers in 2026.

**Key Differentiator:** Built-in evaluation pipeline with LLM-as-judge scoring, golden dataset comparison, and MLflow experiment tracking. This is what hiring managers actually look for.

## Architecture

```
                   ┌──────────────┐
                   │  PDF / Text  │
                   │   Upload     │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  Recursive   │
                   │  Chunking    │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐     ┌────────────────┐
                   │  Embedding   │────▶│  ChromaDB      │
                   │ (all-MiniLM) │     │  Vector Store  │
                   └──────────────┘     └───────┬────────┘
                                                │
                          ┌─────────────────────┘
                          ▼
                   ┌──────────────┐     ┌────────────────┐
                   │  Similarity  │────▶│  LLM Gen       │
                   │  Search      │     │  (qwen3:4b)    │
                   └──────────────┘     └───────┬────────┘
                                                │
                          ┌─────────────────────┘
                          ▼
                   ┌──────────────┐
                   │  Streaming   │
                   │  Response +  │
                   │  Sources     │
                   └──────────────┘

                   ┌──────────────┐     ┌────────────────┐
                   │  Evaluation  │────▶│  MLflow        │
                   │  Judge       │     │  Tracking      │
                   │ (LLM-as-     │     │  (params +     │
                   │  judge)      │     │   metrics)     │
                   └──────────────┘     └────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/SaintChris/rag-eval-system.git
cd rag-eval-system/backend

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Seed data (indexes professional background into vector store)
python3 seed.py

# Run
uvicorn app:app --port 8000

# Open
open http://localhost:8000  # if serving frontend from backend
# or use the frontend/ files separately
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend | FastAPI | Async, fast, auto-docs |
| Vector Store | ChromaDB | Local-first, zero config |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | 80MB, fast on CPU, good quality |
| Generation | qwen3:4b (Ollama) | 2.5GB local, no API keys |
| Experiment Tracking | MLflow | Industry standard, param + metric logging |
| Evaluation | LLM-as-judge (qwen3:1.7b) | Automated faithfulness + scoring |
| Frontend | Vanilla HTML/CSS/JS | Zero build step, no framework |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/index-text` | Index raw text with chunk params |
| `POST` | `/index-pdf` | Upload and index a PDF |
| `POST` | `/query` | Streaming Q&A with source citations |
| `POST` | `/evaluate` | Run eval suite, log metrics to MLflow |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What trading rules does Alex use?"}'
```

### Example: Evaluate

```bash
curl -X POST http://localhost:8000/evaluate \
  -F "chunk_size=300" \
  -F "chunk_overlap=50"
```

## Eval Metrics

| Metric | What It Measures | How | Target |
|--------|-----------------|-----|--------|
| **Faithfulness** | Is the answer grounded in retrieved context? | LLM-as-judge (1-5) | >4.0 |
| **Relevance** | Does the answer match ground truth? | LLM-as-judge (1-5) | >4.0 |
| **Retrieval Accuracy** | Are key terms present in retrieved chunks? | Keyword overlap | >85% |

The **Golden Dataset** (`eval/golden_dataset.json`) contains 3 QA pairs covering professional background, flagship projects, and trading rules. Extend it with domain-specific questions for deeper evaluation.

## Project Structure

```
rag-eval-system/
├── backend/
│   ├── rag_engine.py      # Chunking + ChromaDB + embeddings
│   ├── llm_client.py      # Ollama LLM with streaming
│   ├── tracker.py         # MLflow experiment tracking
│   ├── app.py             # FastAPI routes
│   ├── seed.py            # One-time data seeding
│   ├── run.py             # Orchestrator (MLflow server + FastAPI)
│   └── requirements.txt
├── eval/
│   ├── evaluator.py       # LLM-as-judge scoring
│   └── golden_dataset.json # QA pairs for evaluation
├── frontend/
│   ├── index.html         # Dashboard UI
│   ├── style.css          # Dark theme
│   └── app.js             # Chat + eval controls
└── tests/
    ├── test_app.py
    ├── test_rag_engine.py
    ├── test_llm_client.py
    ├── test_evaluator.py
    └── test_tracker.py
```

## Design Decisions

**Why sentence-transformers instead of embeddings?**
Ollama v0.30.7's `/api/embeddings` endpoint has response-time issues on M1 Mac for both nomic-embed-text (274MB) and all-minilm (16MB). Sentence-transformers runs the same all-MiniLM-L6-v2 model locally in-process with no network hop. First call loads the model (~2s), subsequent embeddings are ~18ms each.

**Why local-only?**
This project demonstrates production-grade AI engineering skill without requiring cloud credits. Every component runs on a 16GB M1 Mac. The architecture is portable to AWS/GCP by swapping the tracking URI and model hosting.

**Why evaluation matters?**
Anyone can build a RAG that answers questions. Measuring *whether those answers are correct* is what separates senior engineers from juniors. This project builds that muscle.

## Extending

**Add more domain data:** Edit `backend/seed.py` to add your own documents, then re-run `python3 seed.py`.

**Add API authentication:** Add middleware to `app.py`:
```python
from fastapi.security import HTTPBearer
```

**Scale to cloud:** Change `MLflowTracker` tracking_uri to a remote server and swap `LocalEmbeddings` for a hosted embedding API.

**Add more eval questions:** Extend `eval/golden_dataset.json` with domain-specific QA pairs.

## License

MIT
