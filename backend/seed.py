"""
Seed the RAG vector store with Alex's professional background data.
Run: cd ~/github/rag-eval-system/backend && source venv/bin/activate && python3 seed.py
"""
import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import RAGEngine

# Clear old DB
db_path = "./chroma_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"Cleared old DB at {db_path}")

engine = RAGEngine(persist_directory=db_path)

documents = [
    {
        "text": """
Alex Bogle is a 27-year-old AI Engineer and IT Operations professional based in Hellshire, Portmore, Jamaica.
He has 4 years of experience in IT operations, managing over 6,000 assets.
His core technical stack includes Python, Docker, PostgreSQL, Prometheus, and Grafana.
He specializes in MLOps, AI infrastructure, and building production-grade AI systems.
Alex is deeply committed to faith, discipline, and maximum autonomy.
""",
        "metadata": {"source": "bio", "category": "background"}
    },
    {
        "text": """
Alex's flagship project is a 6-agent multi-agent AI system built entirely on his M1 Mac (16GB RAM)
using free tools and zero cloud cost. The system handled over 22,000 requests, has 52 automated test cases,
and runs with $0 cloud spend. Workers use LangChain, Ollama (qwen3:4b), and coordinate through
a custom orchestration layer. This demonstrates production-grade AI engineering without enterprise budgets.
""",
        "metadata": {"source": "multi-agent-project", "category": "projects"}
    },
    {
        "text": """
Alex built a simple portfolio website at saintlex.sbs using GitHub Pages, showcasing his
projects, skills, and background. The site includes project links, contact information,
and a professional layout optimized for recruiters and hiring managers.
""",
        "metadata": {"source": "portfolio", "category": "projects"}
    },
    {
        "text": """
Alex has hands-on experience with AI/ML tools including Cursor AI, Claude, GitHub Copilot,
and Hermes Agent on a daily basis. He uses these tools for coding, system administration,
trading analysis, and content creation. He has practical experience with LLM APIs, prompt engineering,
RAG pipelines, fine-tuning workflows, and ML experiment tracking with MLflow.
""",
        "metadata": {"source": "ai-tools", "category": "skills"}
    },
    {
        "text": """
Alex's trading approach uses liquidity-based market structure analysis (SMC + ICT influence).
His execution model: Sweep, Displacement, Retest, Continuation.
He trades MNQ1 only with 0.25 contract size and a tick value of $0.50.
Stop loss range: 200-334 ticks (never under 150). Target: minimum 1.5:1 to 2:1 risk-reward ratio.
Maximum 3 trades per day. Stops trading after 2 consecutive losses. Maximum daily loss: $500.
Position holding time: 1-2 hours. Primary trading window: 11:00 AM ET (shorts).
Secondary windows: 6:00 PM and 11:00 PM ET (longs).
""",
        "metadata": {"source": "trading", "category": "trading"}
    },
    {
        "text": """
Alex is currently pursuing remote AI/ML engineering roles targeting MLOps Engineer, AI Engineer,
AI Infrastructure Engineer, and LLM Engineer positions. His differentiator is building production
AI systems with zero cloud budget, demonstrating resourcefulness and deep technical understanding.
""",
        "metadata": {"source": "career", "category": "career"}
    },
]

print(f"Indexing {len(documents)} documents into ChromaDB (using all-minilm embeddings)...")
for i, doc in enumerate(documents, 1):
    engine.index_text(
        doc["text"],
        chunk_size=300,
        chunk_overlap=50,
        metadata=doc["metadata"]
    )
    print(f"  [{i}/{len(documents)}] Indexed: {doc['metadata']['source']}")

# Verify
results = engine.retrieve("What is Alex's trading strategy?", k=2)
print(f"\nVerification query 'What is Alex's trading strategy?' returned {len(results)} results")
for r in results:
    print(f"  -> {r.page_content[:80]}...")

print(f"\nSeed complete. {len(documents)} documents indexed.")
