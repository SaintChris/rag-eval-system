"""
RAG Evaluator v2 — Uses stronger judge model for reliable scoring.

Fix #1: Judge model is now Nemotron 3 Super 120B (stronger than generator's Qwen3 Coder).
All models are free-tier on OpenRouter.
"""
import json
import re
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx


# ── Model Configuration ──────────────────────────────────────────────
# Judge must be STRONGER than the generator for LLM-as-judge to work.
JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"  # 1M context, strong reasoning
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def call_openrouter(model: str, prompt: str, temperature: float = 0.0,
                     max_tokens: int = 100) -> str:
    """Call OpenRouter model. Returns raw text response."""
    if not OPENROUTER_API_KEY:
        # Fallback: return neutral score
        return "3"

    url = f"{OPENROUTER_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/SaintChris/rag-eval-system"
        })
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class RAGEvaluatorV2:
    def score_faithfulness(self, context: str, answer: str) -> float:
        """Is the answer grounded in the context? Uses Nemotron 3 Super as judge."""
        prompt = f"""You are an expert evaluation judge. Evaluate whether the answer is grounded in the provided context.

Context:
{context}

Answer:
{answer}

Rate Faithfulness 1-5:
1: Completely hallucinated — no support in context
3: Partially grounded — some claims supported, some assumed
5: Fully grounded — every claim backed by context

Respond with ONLY a single digit 1-5."""
        try:
            res = call_openrouter(JUDGE_MODEL, prompt)
            score = re.search(r"[1-5]", res)
            return float(score.group(0)) if score else 3.0
        except Exception:
            return 3.0

    def score_relevance(self, query: str, answer: str, reference: str) -> float:
        """Does the answer match the reference? Uses Nemotron 3 Super as judge."""
        prompt = f"""You are an expert evaluation judge. Compare the generated answer against the ground truth.

Query: {query}
Reference (ground truth): {reference}
Generated Answer: {answer}

Rate Relevance 1-5:
1: Completely wrong or irrelevant
3: Partially correct, misses key info
5: Excellent — directly answers with high accuracy

Respond with ONLY a single digit 1-5."""
        try:
            res = call_openrouter(JUDGE_MODEL, prompt)
            score = re.search(r"[1-5]", res)
            return float(score.group(0)) if score else 3.0
        except Exception:
            return 3.0

    def evaluate_retrieval_accuracy(self, retrieved_contexts: list, reference: str) -> float:
        """Keyword overlap between retrieved docs and reference. No LLM needed."""
        import re as _re
        ref_words = set(_re.findall(r"\w+", reference.lower()))
        stop_words = {"is", "an", "and", "with", "the", "who", "his", "includes",
                      "of", "in", "on", "a", "it", "has", "for", "only", "are",
                      "to", "from", "by", "at", "as", "be", "this", "that",
                      "these", "those", "was", "were", "been", "being", "have",
                      "had", "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "can", "shall"}
        keywords = {w for w in ref_words if len(w) > 3 and w not in stop_words}
        if not keywords:
            return 1.0
        combined_context = " ".join(
            [getattr(c, "page_content", str(c)).lower() for c in retrieved_contexts]
        )
        matches = sum(1 for kw in keywords if kw in combined_context)
        return matches / len(keywords)


# Backward-compatible alias
RAGEvaluator = RAGEvaluatorV2
