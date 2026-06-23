"""Test the RAG system with rate-limit-aware config."""
import os, sys, json, time, httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
from rag_engine_v2 import RAGEngineV2

engine = RAGEngineV2(persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"))

QUERIES = [
    "What is Alex's trading strategy?",
    "What is Alex's five-year plan?",
    "What projects has Alex built?",
]

results = []
for qi, query in enumerate(QUERIES):
    print(f"\n{'='*60}")
    print(f"Query {qi+1}: {query}")
    print(f"{'='*60}")

    start = time.time()
    docs = engine.retrieve(query, k=3, use_expansion=False)
    answer = "".join(engine.generate_answer(query, docs, stream=False))
    elapsed = time.time() - start

    print(f"Docs: {len(docs)} | Time: {elapsed:.1f}s | Chars: {len(answer)} | Citations: {'[' in answer}")
    print(f"\n{answer[:300]}")

    results.append({
        "query": query,
        "answer": answer,
        "docs": len(docs),
        "time": round(elapsed, 1),
        "citations": "[" in answer,
    })

    # Rate-limit spacing between queries
    if qi < len(QUERIES) - 1:
        time.sleep(5)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for r in results:
    status = "OK" if r["answer"] and not r["answer"].startswith("ERROR") else "FAIL"
    print(f"  [{status}] {r['query'][:40]:40} | {r['time']}s | {len(r['answer'])} chars | citations: {r['citations']}")
