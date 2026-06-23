"""Quick test with fallback model config."""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
from rag_engine_v2 import RAGEngineV2

engine = RAGEngineV2(persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"))

query = "What is Alex's trading strategy?"
print(f"Query: {query}")
print(f"Primary model: openrouter/owl-alpha")
print(f"Fallback model: qwen/qwen3-coder:free")
print("-" * 60)

docs = engine.retrieve(query, k=2, use_expansion=False)
print(f"Retrieved {len(docs)} docs")

time.sleep(2)
answer = "".join(engine.generate_answer(query, docs, stream=False))
print(f"\nAnswer ({len(answer)} chars, citations: {'[' in answer}):")
print(answer[:400])
