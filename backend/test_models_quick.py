"""
A/B test free OpenRouter models — quick version, 2 queries only.
"""
import os, sys, json, time, httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")

from rag_engine_v2 import RAGEngineV2

CHROMA_PERSIST = os.path.join(os.path.dirname(__file__), "chroma_db")

MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-coder:free",
    "google/gemma-4-26b-a4b-it:free",
]

QUERIES = [
    "What is Alex's trading strategy and edge?",
    "What is Alex's five-year plan?",
]


def main():
    print("RAG Model A/B Test (quick)")
    print("=" * 60)

    engine = RAGEngineV2(persist_directory=CHROMA_PERSIST)
    all_results = {}

    for qi, query in enumerate(QUERIES):
        print(f"\nQUERY {qi+1}: {query}")
        print("-" * 60)

        for model_id in MODELS:
            import rag_engine_v2
            original = rag_engine_v2.LLM_MODEL
            rag_engine_v2.LLM_MODEL = model_id

            try:
                docs = engine.retrieve(query, k=3, use_expansion=False)
                time.sleep(3)
                answer = "".join(engine.generate_answer(query, docs, stream=False))
                has_cit = "[" in answer

                print(f"\n  {model_id}")
                print(f"  Docs: {len(docs)} | Chars: {len(answer)} | Citations: {has_cit}")
                print(f"  {answer[:250]}")

                all_results.setdefault(model_id, []).append({
                    "query": query,
                    "answer": answer,
                    "docs": len(docs),
                    "citations": has_cit,
                })
            except Exception as e:
                print(f"\n  {model_id}: ERROR - {str(e)[:80]}")
            finally:
                rag_engine_v2.LLM_MODEL = original

            time.sleep(3)  # Rate-limit spacing

    with open("model_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Results saved to model_test_results.json")


if __name__ == "__main__":
    main()
