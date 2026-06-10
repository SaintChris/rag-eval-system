import json
import re
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.llm_client import LLMClient


class RAGEvaluator:
    def __init__(self, judge_model="qwen3:4b"):
        self.llm = LLMClient(model=judge_model)

    def score_faithfulness(self, context: str, answer: str) -> float:
        prompt = f"""
You are an expert evaluation judge. You are evaluating whether a generated answer is grounded in the provided context.
Context:
{context}

Answer:
{answer}

Rate the Faithfulness of the answer on a scale from 1 to 5:
1: The answer contains completely hallucinated facts not in the context.
3: The answer has some grounded facts but adds unverified assumptions.
5: The answer is 100% grounded and fully backed by the context.

Respond with ONLY a single digit score between 1 and 5. Do not write anything else.
Score:"""
        try:
            res = self.llm.generate(prompt, temperature=0.0)
            score = re.search(r"[1-5]", res)
            return float(score.group(0)) if score else 3.0
        except Exception:
            return 3.0

    def score_relevance(self, query: str, answer: str, reference: str) -> float:
        prompt = f"""
You are an expert evaluation judge. You are comparing a generated answer against a ground truth reference for a given query.
Query: {query}
Reference Ground Truth: {reference}
Generated Answer: {answer}

Rate the Relevance and Correctness of the answer on a scale from 1 to 5:
1: Completely irrelevant or wrong.
3: Partially correct but misses key information.
5: Excellent answer, highly accurate and directly answers the query.

Respond with ONLY a single digit score between 1 and 5. Do not write anything else.
Score:"""
        try:
            res = self.llm.generate(prompt, temperature=0.0)
            score = re.search(r"[1-5]", res)
            return float(score.group(0)) if score else 3.0
        except Exception:
            return 3.0

    def evaluate_retrieval_accuracy(self, retrieved_contexts: list, reference: str) -> float:
        ref_words = set(re.findall(r"\w+", reference.lower()))
        stop_words = {"is", "an", "and", "with", "the", "who", "his", "includes", "of", "in", "on", "a", "it", "has", "for", "only"}
        keywords = {w for w in ref_words if len(w) > 3 and w not in stop_words}
        if not keywords:
            return 1.0
        combined_context = " ".join([c.page_content.lower() for c in retrieved_contexts])
        matches = sum(1 for kw in keywords if kw in combined_context)
        return matches / len(keywords)
