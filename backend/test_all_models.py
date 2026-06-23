"""Test all 3 models with a simple query."""
import os, sys, json, time, httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
from rag_engine_v2 import RAGEngineV2

engine = RAGEngineV2(persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"))

MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-coder:free",
    "google/gemma-4-26b-a4b-it:free",
]

for model_id in MODELS:
    import rag_engine_v2
    rag_engine_v2.LLM_MODEL = model_id

    docs = engine.retrieve("What is Alex trading strategy?", k=2, use_expansion=False)
    time.sleep(3)
    answer = "".join(engine.generate_answer("What is Alex trading strategy?", docs, stream=False))

    print(f"\n{model_id}")
    print(f"  Length: {len(answer)} | Citations: {'[' in answer}")
    print(f"  {answer[:200]}")
    time.sleep(5)
