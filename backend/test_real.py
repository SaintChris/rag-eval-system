"""Test RAG with real API key from Hermes .env."""
import os, sys, json, time

hermes_env = os.path.expanduser("~/.hermes/.env")
key = None
with open(hermes_env, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

if not key or "***" in key:
    print("ERROR: Could not read real key from Hermes .env")
    sys.exit(1)

print(f"Key length: {len(key)}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["OPENROUTER_API_KEY"] = key

from rag_engine_v2 import RAGEngineV2

engine = RAGEngineV2(persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"))

query = "What is Alex's trading strategy?"
print(f"\nQuery: {query}")
print("-" * 60)

docs = engine.retrieve(query, k=2, use_expansion=False)
print(f"Retrieved {len(docs)} docs")
for d in docs:
    print(f"  [{d.metadata.get('source', '?')}] {d.page_content[:80]}...")

time.sleep(2)
answer = "".join(engine.generate_answer(query, docs, stream=False))
print(f"\nAnswer ({len(answer)} chars, citations: {'[' in answer}):")
print(answer[:500])
