"""
A/B test free OpenRouter models for RAG generation.
Tests the same query against 3 models and compares answers.
"""
import os
import sys
import json
import time
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")

from rag_engine_v2 import RAGEngineV2, call_openrouter

VAULT_PATH = os.path.expanduser("~/Obsidian Vault")
CHROMA_PERSIST = os.path.join(os.path.dirname(__file__), "chroma_db")

# Models to test (all free)
MODELS = {
    "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron 3 Super 120B",
    "qwen/qwen3-coder:free": "Qwen3 Coder 480B",
    "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B",
}

# Test queries covering different domains
QUERIES = [
    "What is Alex's trading strategy and edge?",
    "What projects has Alex built?",
    "What is Alex's faith and how does it guide his life?",
    "What mistakes has Alex made in trading?",
    "What is Alex's five-year plan?",
]


def test_model(model_id, model_name, query, engine):
    """Test a single model with a single query."""
    # Temporarily override the LLM model
    import rag_engine_v2
    original_model = rag_engine_v2.LLM_MODEL
    rag_engine_v2.LLM_MODEL = model_id

    try:
        start = time.time()
        docs = engine.retrieve(query, k=3, use_expansion=False)
        time.sleep(1)  # Rate-limit spacing
        answer = "".join(engine.generate_answer(query, docs, stream=False))
        elapsed = time.time() - start

        return {
            "model": model_name,
            "model_id": model_id,
            "answer": answer,
            "num_docs": len(docs),
            "time": round(elapsed, 1),
            "has_citations": "[" in answer and "]" in answer,
            "answer_length": len(answer),
        }
    except Exception as e:
        return {
            "model": model_name,
            "model_id": model_id,
            "answer": f"ERROR: {str(e)[:100]}",
            "num_docs": 0,
            "time": 0,
            "has_citations": False,
            "answer_length": 0,
        }
    finally:
        rag_engine_v2.LLM_MODEL = original_model


def main():
    print("=" * 70)
    print("RAG Model A/B Test — Free OpenRouter Models")
    print("=" * 70)

    engine = RAGEngineV2(persist_directory=CHROMA_PERSIST)

    results = {}

    for qi, query in enumerate(QUERIES):
        print(f"\n{'─' * 70}")
        print(f"QUERY {qi+1}: {query}")
        print(f"{'─' * 70}")

        for model_id, model_name in MODELS.items():
            print(f"\n  Testing {model_name}...")
            result = test_model(model_id, model_name, query, engine)
            results.setdefault(model_name, []).append(result)

            print(f"    Time: {result['time']}s | Docs: {result['num_docs']} | "
                  f"Length: {result['answer_length']} | Citations: {result['has_citations']}")
            print(f"    Answer: {result['answer'][:200]}...")

            # Rate-limit spacing between models
            time.sleep(2)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for model_name, model_results in results.items():
        avg_time = sum(r["time"] for r in model_results) / len(model_results)
        avg_length = sum(r["answer_length"] for r in model_results) / len(model_results)
        citation_rate = sum(1 for r in model_results if r["has_citations"]) / len(model_results) * 100
        errors = sum(1 for r in model_results if r["answer"].startswith("ERROR"))

        print(f"\n  {model_name}:")
        print(f"    Avg time:     {avg_time:.1f}s")
        print(f"    Avg length:   {avg_length:.0f} chars")
        print(f"    Citation rate: {citation_rate:.0f}%")
        print(f"    Errors:       {errors}/{len(model_results)}")

    # Save detailed results
    output = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_queries": len(QUERIES),
        "num_models": len(MODELS),
        "results": {
            name: [
                {k: v for k, v in r.items() if k != "model_id"}
                for r in model_results
            ]
            for name, model_results in results.items()
        },
    }

    with open("model_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to model_test_results.json")


if __name__ == "__main__":
    main()
