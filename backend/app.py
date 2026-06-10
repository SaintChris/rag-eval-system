import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_engine import RAGEngine
from backend.llm_client import LLMClient
from backend.tracker import MLflowTracker
from eval.evaluator import RAGEvaluator

app = FastAPI(title="RAG Eval & Tracking System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

persist_dir = "./chroma_db"
rag = RAGEngine(persist_directory=persist_dir)
llm = LLMClient()
tracker = MLflowTracker(experiment_name="rag-eval-system")
evaluator = RAGEvaluator()


class QueryRequest(BaseModel):
    query: str
    chunk_size: int = 500
    chunk_overlap: int = 50


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/index-text")
def index_text_endpoint(text: str = Form(...), chunk_size: int = Form(500), chunk_overlap: int = Form(50)):
    try:
        rag.index_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return {"status": "success", "message": f"Indexed custom text with chunk size {chunk_size}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-pdf")
async def index_pdf_endpoint(file: UploadFile = File(...), chunk_size: int = Form(500), chunk_overlap: int = Form(50)):
    try:
        os.makedirs("./temp", exist_ok=True)
        temp_path = f"./temp/{file.filename}"
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        rag.index_pdf(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        os.remove(temp_path)
        return {"status": "success", "message": f"Indexed {file.filename}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    docs = rag.retrieve(req.query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    prompt = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {req.query}
Helpful Answer:"""

    def stream_response():
        yield f"__METADATA__:{json.dumps(sources)}\n"
        for chunk in llm.generate_stream(prompt, system_prompt="Answer directly and truthfully."):
            yield chunk

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.post("/evaluate")
def evaluate_endpoint(chunk_size: int = Form(500), chunk_overlap: int = Form(50)):
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "eval", "golden_dataset.json")
    if not os.path.exists(dataset_path):
        dataset_path = "eval/golden_dataset.json"
    with open(dataset_path, "r") as f:
        golden_set = json.load(f)

    results = []
    total_faithfulness = 0
    total_relevance = 0
    total_retrieval_acc = 0

    with tracker.start_run(run_name=f"eval_cs_{chunk_size}_co_{chunk_overlap}"):
        for item in golden_set:
            q = item["question"]
            ref = item["reference"]
            docs = rag.retrieve(q, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
            generated_answer = llm.generate(prompt)
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

        num_questions = len(golden_set)
        avg_faithfulness = total_faithfulness / num_questions
        avg_relevance = total_relevance / num_questions
        avg_retrieval_acc = total_retrieval_acc / num_questions

        tracker.log_params({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": "nomic-embed-text",
            "llm_model": "qwen3:4b",
            "eval_size": num_questions
        })
        tracker.log_metrics({
            "faithfulness": avg_faithfulness,
            "relevance": avg_relevance,
            "retrieval_accuracy": avg_retrieval_acc
        })

    return {
        "status": "success",
        "metrics": {
            "faithfulness": avg_faithfulness,
            "relevance": avg_relevance,
            "retrieval_accuracy": avg_retrieval_acc
        },
        "details": results
    }
