"""
Seed the RAG vector store with Alex's actual data.
Run: cd /Users/alex/rag-eval-system/backend && source .venv/bin/activate && python3 seed_v2.py
"""
import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")

from rag_engine_v2 import RAGEngineV2

db_path = "./chroma_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"Cleared old DB")

engine = RAGEngineV2(persist_directory=db_path)

documents = [
    {
        "text": """
Alex Benjamin St. Christopher Bogle is a 27-year-old AI/ML engineer and futures trader from Hellshire, Portmore, Jamaica.
He runs Hermes Agent — a multi-agent AI system built on a MacBook M1 with zero cloud cost.
His core technical stack includes Python, FastAPI, Next.js, Docker, ChromaDB, Ollama, and OpenRouter.
He specializes in MLOps, AI infrastructure, RAG pipelines, and building production-grade AI systems.
""",
        "metadata": {"source": "bio", "category": "background"}
    },
    {
        "text": """
Alex's trading approach uses liquidity-based market structure analysis (SMC + ICT influence).
His execution model: Sweep, Displacement, Retest, Continuation.
He trades MNQ1 (NASDAQ-100 futures) with a 73% win rate and +$10,352 profit.
His rules: Stop loss 200-334 ticks (never under 150). Target: minimum 1.5:1 to 2:1 risk-reward ratio.
Maximum 3 trades per day. Stops trading after 2 consecutive losses. Maximum daily loss: $500.
Position holding time: 1-2 hours. Primary trading window: 11:00 AM ET (shorts).
He is banned from ES/MES after -$5,318 loss.
Prop firms: Apex (5 accounts, +$580, 0 payouts), FundedNext (9 accounts, 283 trades, -$342).
Monthly income target: $9,600-$16,032.
""",
        "metadata": {"source": "trading", "category": "trading"}
    },
    {
        "text": """
Alex's flagship project is a 6-agent multi-agent AI system built entirely on a local M1 Mac (16GB RAM)
using free tools and zero cloud cost. The system handles research, engineering, data analysis, psychology,
biblical scholarship, performance coaching, and communication. It runs 8 cron jobs autonomously.
Workers use LangChain, Ollama (qwen3:4b), and coordinate through a custom orchestration layer.
This demonstrates production-grade AI engineering without enterprise budgets.
""",
        "metadata": {"source": "multi-agent-project", "category": "projects"}
    },
    {
        "text": """
Alex built a RAG evaluation system using FastAPI + ChromaDB + sentence-transformers + Ollama + MLflow.
It was published on Dev.to as "I Built a Production RAG System on My M1 Mac for $0".
The v2 upgrade added: cross-encoder re-ranking, query expansion, stronger judge model (Nemotron 3 Super),
adaptive chunking, conversation memory, source citation, and Pinecone support.
GitHub: github.com/SaintChris/rag-eval-system
""",
        "metadata": {"source": "rag-project", "category": "projects"}
    },
    {
        "text": """
Alex built a Wallet Budget App using Next.js 16 + TypeScript + Tailwind + Supabase + OpenRouter AI.
12 pages: landing, login, register, dashboard, transactions, budgets, goals, import, AI assistant.
Features CSV/Excel/PDF bank statement parsing for Jamaican banks.
GitHub: SaintChris/budgetwise (private). Vercel deployed but blocked by Deployment Protection SSO.
""",
        "metadata": {"source": "budget-app", "category": "projects"}
    },
    {
        "text": """
Alex is currently pursuing remote AI/ML engineering roles targeting MLOps Engineer, AI Engineer,
AI Infrastructure Engineer, and LLM Engineer positions. His differentiator is building production
AI systems with zero cloud budget, demonstrating resourcefulness and deep technical understanding.
He has 68+ tracked job applications. He completed Week 1 of a CrewAI Udemy course.
His 5 portfolio projects: RAG system (done), Chroma contribution (done), Transformer fine-tuning,
MLOps pipeline, and end-to-end ML application.
""",
        "metadata": {"source": "career", "category": "career"}
    },
    {
        "text": """
Alex is Ethiopian Orthodox Christian. He reads the 81-book Ethiopian Orthodox canon including
the books of Enoch, Jubilees, the Meqabyan, and the broader Deuterocanon.
His faith is central to his life. He prays, fasts, and seeks God's face daily.
He believes his trading calling is from God and that financial independence is part of his five-year plan
to move from Hellshire to Montego Bay to freedom.
""",
        "metadata": {"source": "faith", "category": "faith"}
    },
]

print(f"Indexing {len(documents)} documents about Alex...\n")
for i, doc in enumerate(documents, 1):
    count = engine.index_text(
        doc["text"],
        content_type=doc["metadata"].get("category", "general"),
        metadata=doc["metadata"]
    )
    print(f"  [{i}/{len(documents)}] {doc['metadata']['source']}: {count} chunks")

# Verify
print("\n--- Verification ---")
test_queries = [
    "What is Alex's trading strategy?",
    "What projects has Alex built?",
    "What is Alex's faith?",
]
for q in test_queries:
    docs = engine.retrieve(q, k=2)
    answer = "".join(engine.generate_answer(q, docs, stream=False))
    print(f"\nQ: {q}")
    print(f"A: {answer[:200]}...")

print(f"\nSeed complete. {len(documents)} documents indexed.")
