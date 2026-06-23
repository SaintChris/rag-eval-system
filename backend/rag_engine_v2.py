"""
RAG Engine v2 — Production-grade with all fixes applied.

Fixes:
1. Re-ranking with cross-encoder
2. Query expansion
3. Empty retrieval guard
4. Source citation
5. Conversation memory
6. Adaptive chunking
7. Dynamic model selection via OpenRouter
"""
import os
import json
import time
import httpx
import numpy as np
from typing import Optional, List, Dict, Any, Generator, Union
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ── Model Configuration ──────────────────────────────────────────────
# All models are free-tier on OpenRouter, selected per-task.

EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # Local, fast, good quality
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Local re-ranker

# OpenRouter models — API key set via OPENROUTOR_API_KEY env var
LLM_MODEL = "openrouter/owl-alpha"           # Primary: your paid model, reliable
LLM_FALLBACK_MODEL = "qwen/qwen3-coder:free"  # Fallback: free, good quality
JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"  # Evaluation judge — stronger than generator
QUERY_EXPAND_MODEL = "google/gemma-4-26b-a4b-it:free"   # Query expansion — fast, creative

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
def _load_openrouter_key() -> str:
    """Load OpenRouter key from Hermes .env or local .env."""
    # Try Hermes .env first (used by gateway)
    hermes_env = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(hermes_env):
        with open(hermes_env) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key and "***" not in key:
                        return key
    # Try local .env
    local_env = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(local_env):
        with open(local_env) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key and "***" not in key:
                        return key
    return os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_API_KEY = _load_openrouter_key()
if OPENROUTER_API_KEY:
    print(f"[engine] OpenRouter key loaded: {OPENROUTER_API_KEY[:10]}... (len={len(OPENROUTER_API_KEY)})")
else:
    print("[engine] WARNING: No OpenRouter key found!")


# ── OpenRouter Client ───────────────────────────────────────────────

# Rate-limit tracking per model
_rate_limit_state: dict[str, dict] = {}


def _wait_for_rate_limit(model: str, min_interval: float = 3.0) -> None:
    """Ensure minimum interval between calls to the same model."""
    now = time.time()
    state = _rate_limit_state.get(model, {})
    last_call = state.get("last_call", 0.0)
    wait = min_interval - (now - last_call)
    if wait > 0:
        time.sleep(wait)
    _rate_limit_state[model] = {"last_call": time.time()}


def call_openrouter(model: str, prompt: str, system_prompt: str = "",
                     temperature: float = 0.2, max_tokens: int = 2048,
                     retries: int = 3, fallback_model: str = "") -> str:
    """Call any OpenRouter model with rate-limit spacing and retry."""
    global _rate_limit_state

    if not OPENROUTER_API_KEY:
        return call_ollama(prompt, system_prompt, temperature)

    # Per-model rate limiting (3s between calls to same model)
    _wait_for_rate_limit(model, min_interval=3.0)

    url = f"{OPENROUTER_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    for attempt in range(retries):
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://github.com/SaintChris/rag-eval-system"
            })
            if resp.status_code == 429:
                # Try fallback model if available
                if fallback_model and model != fallback_model:
                    print(f"[openrouter] Rate limited ({model}), trying fallback ({fallback_model})...")
                    fallback_payload = dict(payload)
                    fallback_payload["model"] = fallback_model
                    with httpx.Client(timeout=120.0) as fb_client:
                        fb_resp = fb_client.post(url, json=fallback_payload, headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "HTTP-Referer": "https://github.com/SaintChris/rag-eval-system"
                        })
                        if fb_resp.status_code == 200:
                            content = fb_resp.json()["choices"][0]["message"]["content"].strip()
                            _rate_limit_state[model] = {"last_call": time.time()}
                            return content
                        elif fb_resp.status_code == 429:
                            print(f"[openrouter] Fallback also rate limited, retrying in {wait}s...")
                # Exponential backoff: 5s, 10s, 20s
                wait = 5 * (2 ** attempt)
                print(f"[openrouter] Rate limited ({model}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            _rate_limit_state[model] = {"last_call": time.time()}
            return content
    # Fallback to Ollama after retries exhausted
    print(f"[openrouter] Retries exhausted ({model}), falling back to Ollama")
    return call_ollama(prompt, system_prompt, temperature)


def call_ollama(prompt: str, system_prompt: str = "", temperature: float = 0.2,
                model: str = "qwen3:4b") -> str:
    """Fallback: local Ollama."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt or "You are a helpful assistant.",
        "options": {"temperature": temperature},
        "stream": False
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


# ── Local Embeddings (unchanged — runs offline, no API cost) ─────────

class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"[embeddings] Loading: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"[embeddings] Dim: {self.model.get_sentence_embedding_dimension()}")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


# ── Query Expansion ──────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """Generate reformulations to improve recall. Uses Gemma 4 26B for speed."""
    prompt = f"""Given this query, generate 2 alternative phrasings that capture the same intent.
Use different vocabulary and angles. One should be more specific, one more general.

Query: {query}

Format as JSON array: ["rephrasing1", "rephrasing2"]
Return ONLY the JSON array."""
    try:
        result = call_openrouter(QUERY_EXPAND_MODEL, prompt, temperature=0.5, max_tokens=200)
        # Extract JSON array from response
        start = result.find("[")
        end = result.rfind("]") + 1
        if start >= 0 and end > start:
            variants = json.loads(result[start:end])
            return [query] + variants[:2]  # Original + 2 variants
    except Exception:
        pass
    return [query]  # Fallback: original only


# ── Re-ranking ───────────────────────────────────────────────────────

class ReRanker:
    def __init__(self, model_name: str = RERANK_MODEL):
        print(f"[reranker] Loading: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[str], top_k: int = 3) -> list[tuple[str, float]]:
        """Re-rank documents by cross-encoder score. Returns (doc, score) pairs."""
        if not docs:
            return []
        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = list(zip(docs, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ── Adaptive Chunking ───────────────────────────────────────────────

def get_chunk_params(content_type: str = "default") -> tuple[int, int]:
    """Return (chunk_size, chunk_overlap) based on content type."""
    params = {
        "trading": (800, 100),      # Strategies need context
        "technical": (600, 80),      # Code/docs — medium chunks
        "biography": (400, 50),     # Bios — smaller is fine
        "general": (500, 50),        # Default
    }
    return params.get(content_type, params["general"])


# ── RAG Engine v2 ────────────────────────────────────────────────────

class RAGEngineV2:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = LocalEmbeddings()
        self.reranker = ReRanker()
        self.vector_store = None
        self.conversation_history: list[dict] = []  # Fix #5: conversation memory
        self._init_db()

    def _init_db(self):
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"[engine] Loaded existing DB from {self.persist_directory}")

    # ── Indexing ──

    def index_text(self, text: str, content_type: str = "default",
                   metadata: dict = None) -> int:
        chunk_size, chunk_overlap = get_chunk_params(content_type)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        documents = [Document(page_content=c, metadata=metadata or {}) for c in chunks]
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"[index] {len(chunks)} chunks indexed (type={content_type}, size={chunk_size})")
        return len(chunks)

    def index_pdf(self, file_path: str, content_type: str = "default") -> int:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        return self.index_text(
            full_text, content_type=content_type,
            metadata={"source": os.path.basename(file_path), "type": "pdf"}
        )

    def index_markdown(self, md_text: str, source: str = "",
                       content_type: str = "general") -> int:
        return self.index_text(
            md_text, content_type=content_type,
            metadata={"source": source or "markdown", "type": "markdown"}
        )

    # ── Retrieval with all fixes ──

    def retrieve(self, query: str, k: int = 5, use_expansion: bool = True,
                 content_type: str = None) -> list[Document]:
        """
        Full retrieval pipeline:
        1. Query expansion (Fix #4)
        2. Vector search with expanded queries
        3. Re-ranking (Fix #2)
        4. Empty guard (Fix #3)
        """
        if not self.vector_store:
            return []

        # Step 1: Query expansion (with rate-limit delay)
        if use_expansion:
            time.sleep(1)  # Rate-limit spacing
            queries = expand_query(query)
        else:
            queries = [query]

        # Step 2: Retrieve from all expanded queries
        all_docs = []
        seen_content = set()
        for q in queries:
            docs = self.vector_store.similarity_search(q, k=k)
            for doc in docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    all_docs.append(doc)

        if not all_docs:
            return []  # Fix #3: empty guard at retrieval level

        # Step 3: Re-rank
        doc_texts = [doc.page_content for doc in all_docs]
        reranked = self.reranker.rerank(query, doc_texts, top_k=min(k, 3))

        # Map back to Document objects with scores
        text_to_doc = {doc.page_content: doc for doc in all_docs}
        final_docs = []
        for text, score in reranked:
            doc = text_to_doc.get(text)
            if doc:
                doc.metadata["rerank_score"] = float(score)
                final_docs.append(doc)

        return final_docs

    # ── Generation with source citation ──

    def generate_answer(self, query: str, docs: list[Document],
                        stream: bool = True):
        """
        Generate answer with source citations (Fix #8).
        Uses owl-alpha for generation — reliable, strong reasoning.
        """
        if not docs:
            # Fix #3: empty retrieval guard at generation level
            result = "I don't have enough context to answer this. Try rephrasing or indexing more documents."
            if stream:
                yield result
            return result

        # Build numbered context with source labels
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", f"doc-{i}")
            context_parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        # Include conversation memory (Fix #5)
        history_str = ""
        if self.conversation_history:
            recent = self.conversation_history[-4:]  # Last 2 turns
            history_str = "\n\n## Recent conversation:\n"
            for turn in recent:
                role = turn["role"]
                content = turn["content"]
                history_str += f"**{role}**: {content}\n"

        prompt = f"""Answer using ONLY the sources below. If the answer cannot be found, say "I don't know."

Rules:
- Every claim MUST be cited with [1], [2], etc.
- Be direct and concise.
- Do not make up information.

{history_str}

## Sources:
{context}

## Question:
{query}

## Answer (with citations):
"""
        system = "You are a precise RAG assistant. Every claim must be cited with [1], [2], etc."

        if OPENROUTER_API_KEY:
            result = call_openrouter(LLM_MODEL, prompt, system, temperature=0.2, max_tokens=2048,
                                     fallback_model=LLM_FALLBACK_MODEL)
            print(f"[generate] Model={LLM_MODEL}, Result len={len(result)}, Result[:50]={repr(result[:50])}")
        else:
            result = call_ollama(prompt, system, temperature=0.2)
            print(f"[generate] Ollama, Result len={len(result)}")

        if stream:
            yield result
        else:
            return result

    def generate_answer_sync(self, query: str, docs: list[Document]) -> str:
        """Non-streaming version — returns string directly."""
        if not docs:
            return "I don't have enough context to answer this. Try rephrasing or indexing more documents."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", f"doc-{i}")
            context_parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        history_str = ""
        if self.conversation_history:
            recent = self.conversation_history[-4:]
            history_str = "\n\n## Recent conversation:\n"
            for turn in recent:
                history_str += f"**{turn['role']}**: {turn['content']}\n"

        prompt = f"""Answer using ONLY the sources below. If the answer cannot be found, say "I don't know."

Rules:
- Every claim MUST be cited with [1], [2], etc.
- Be direct and concise.
- Do not make up information.

{history_str}

## Sources:
{context}

## Question:
{query}

## Answer (with citations):
"""
        system = "You are a precise RAG assistant. Every claim must be cited with [1], [2], etc."

        if OPENROUTER_API_KEY:
            result = call_openrouter(LLM_MODEL, prompt, system, temperature=0.2, max_tokens=2048,
                                     fallback_model=LLM_FALLBACK_MODEL)
            print(f"[generate] Model={LLM_MODEL}, Result len={len(result)}, Result[:50]={repr(result[:50])}")
        else:
            result = call_ollama(prompt, system, temperature=0.2)
            print(f"[generate] Ollama, Result len={len(result)}")
        return result

    def query(self, query: str, k: int = 3,
              use_expansion: bool = True) -> dict:
        """Full RAG pipeline: retrieve + generate."""
        docs = self.retrieve(query, k=k, use_expansion=use_expansion)
        time.sleep(1)  # Rate-limit spacing before generation
        answer = self.generate_answer_sync(query, docs)
        print(f"[query] Answer len={len(answer)}, Answer[:50]={repr(answer[:50])}")

        # Update conversation memory (Fix #5)
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer[:500]})

        sources = [{
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in docs]

        return {"answer": answer, "sources": sources, "num_docs": len(docs)}


# ── Backward-compatible alias ───────────────────────────────────────

RAGEngine = RAGEngineV2
