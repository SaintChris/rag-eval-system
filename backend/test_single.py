"""Quick single-query test for one model."""
import os, sys, json, time, httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
from rag_engine_v2 import RAGEngineV2

engine = RAGEngineV2(persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--query", required=True)
args = parser.parse_args()

import rag_engine_v2
rag_engine_v2.LLM_MODEL = args.model

print(f"Model: {args.model}")
print(f"Query: {args.query}")
print("-" * 60)

docs = engine.retrieve(args.query, k=3, use_expansion=False)
print(f"Retrieved {len(docs)} docs")

time.sleep(3)
answer = "".join(engine.generate_answer(args.query, docs, stream=False))
print(f"\nAnswer ({len(answer)} chars, citations: {'[' in answer}):")
print(answer)
