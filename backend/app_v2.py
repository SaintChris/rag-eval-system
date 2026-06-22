"""
RAG API v2 — Production-grade with all fixes applied.

Fixes:
1. Stronger judge model (Nemotron 3 Super 120B vs Qwen3 Coder generator)
2. Re-ranking with cross-encoder
3. Empty retrieval guard (no hallucination on missing docs)
4. Source citation in every answer
5. Conversation memory across turns
6. Adaptive chunking by content type
7. Query expansion for better recall
8. Pinecone support (optional, via PINECONE_API_KEY)
9. OpenRouter model selection per task
10. Streaming via OpenRouter with Ollama fallback
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine_v2 import RAGEngineV2
from evaluator_v2 import RAGEvaluatorV2
from backend.tracker import MLflowTracker

app = FastAPI(title="RAG Eval & Tracking System v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
persist_dir = os.environ.get("CHROMADB_PERSIST_DIR", "./chroma_db")
rag = RAGEngineV2(persist_directory=persist_dir)
evaluator = RAGEvaluatorV2()
tracker = MLflowTracker(experiment_name="rag-eval-v2")


class QueryRequest(BaseModel):
    query: str
    k: int = 3
    content_type: str = None
    use_expansion: bool = True


@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0"}


@app.post("/index-text")
def index_text_endpoint(
    text: str = Form(...),
    content_type: str = Form("default"),
    chunk_size: int = Form(None),
    chunk_overlap: int = Form(None)
):
    try:
        if chunk_size and chunk_overlap:
            # Manual override
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.documents import Document
            from langchain_community.vectorstores import Chroma
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(text)
            documents = [Document(page_content=c) for c in chunks]
            rag.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=rag.embeddings,
                persist_directory=rag.persist_directory
            )
            return {"status": "success", "chunks": len(chunks), "mode": "manual"}
        else:
            # Adaptive chunking
            count = rag.index_text(text, content_type=content_type)
            return {"status": "success", "chunks": count, "content_type": content_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-pdf")
async def index_pdf_endpoint(
    file: UploadFile = File(...),
    content_type: str = Form("default")
):
    try:
        os.makedirs("./temp", exist_ok=True)
        temp_path = f"./temp/{file.filename}"
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        count = rag.index_pdf(temp_path, content_type=content_type)
        os.remove(temp_path)
        return {"status": "success", "file": file.filename, "chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    try:
        result = rag.query(
            query=req.query,
            k=req.k,
            use_expansion=req.use_expansion
        )

        def stream_response():
            yield f"__METADATA__:{json.dumps(result['sources'])}\n"
            yield result["answer"]

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
def evaluate_endpoint(chunk_size: int = Form(500), chunk_overlap: int = Form(50)):
    try:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "eval", "golden_dataset.json")
        if not os.path.exists(dataset_path):
            dataset_path = "eval/golden_dataset.json"
        with open(dataset_path, "r") as f:
            golden_set = json.load(f)

        results = []
        total_faithfulness = 0
        total_relevance = 0
        total_retrieval_acc = 0

        with tracker.start_run(run_name=f"eval_v2_cs_{chunk_size}_co_{chunk_overlap}"):
            for item in golden_set:
                q = item["question"]
                ref = item["reference"]
                docs = rag.retrieve(q, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])

                if not docs:
                    generated_answer = "I don't have enough context."
                else:
                    generated_answer = "".join(
                        rag.generate_answer(q, docs, stream=False)
                    )

                faithfulness = evaluator.score_faithfulness(context, generated_answer)
                relevance = evaluator.score_relevance(q, generated_answer, ref)
                retrieval_acc = evaluator.evaluate_retrieval_accuracy(docs, ref)

                results.append({
                    "question": q,
                    "answer": generated_answer,
                    "reference": ref,
                    "faithfulness": faithfulness,
                    "relevance": relevance,
                    "retrieval_accuracy": retrieval_acc
                })
                total_faithfulness += faithfulness
                total_relevance += relevance
                total_retrieval_acc += retrieval_acc

            n = len(golden_set)
            tracker.log_params({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": "all-MiniLM-L6-v2",
                "rerank_model": "ms-marco-MiniLM-L-6-v2",
                "llm_model": "qwen/qwen3-coder:free",
                "judge_model": "nvidia/nemotron-3-super-120b-a12b:free",
                "query_expand_model": "google/gemma-4-26b-a4b-it:free",
                "eval_size": n,
                "version": "2.0"
            })
            tracker.log_metrics({
                "faithfulness": total_faithfulness / n,
                "relevance": total_relevance / n,
                "retrieval_accuracy": total_retrieval_acc / n
            })

        return {
            "status": "success",
            "version": "2.0",
            "metrics": {
                "faithfulness": total_faithfulness / n,
                "relevance": total_relevance / n,
                "retrieval_accuracy": total_retrieval_acc / n
            },
            "details": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
def config():
    """Show current model configuration."""
    return {
        "version": "2.0",
        "embedding_model": "all-MiniLM-L6-v2 (local)",
        "rerank_model": "ms-marco-MiniLM-L-6-v2 (local)",
        "llm_model": "qwen/qwen3-coder:free (OpenRouter)",
        "judge_model": "nvidia/nemotron-3-super-120b-a12b:free (OpenRouter)",
        "query_expand_model": "google/gemma-4-26b-a4b-it:free (OpenRouter)",
        "fallback": "Ollama qwen3:4b (local)",
        "vector_store": "ChromaDB (local) / Pinecone (optional)",
        "fixes_applied": [
            "Stronger judge model (Nemotron 3 Super > Qwen3 Coder)",
            "Cross-encoder re-ranking",
            "Empty retrieval guard (no hallucination)",
            "Source citation in answers",
            "Conversation memory across turns",
            "Adaptive chunking by content type",
            "Query expansion for recall",
            "OpenRouter model selection per task",
            "Streaming with Ollama fallback",
            "Pinecone support (optional)"
        ]
    }
