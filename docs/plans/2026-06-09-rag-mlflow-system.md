# RAG Document Q&A with MLflow Tracking Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a robust, local RAG pipeline using LangChain, ChromaDB, and Ollama that tracks hyperparameters and evaluation metrics inside MLflow, serving as a high-signal portfolio piece.

**Architecture:**
1. RAG pipeline built with LangChain, using ChromaDB as the vector store and `nomic-embed-text` via Ollama for embeddings.
2. FastAPI backend exposing endpoints for document indexing, QA streaming, and running eval suites.
3. MLflow Tracking integration to record RAG runs (chunk size, overlap, prompt template, model) and aggregate evaluation metrics (Faithfulness, Relevance, Retrieval Accuracy).
4. Local evaluations powered by an LLM-as-judge (using Qwen-3) scoring a Golden Dataset of Alex's professional background and project context.
5. Simple, responsive HTML/JS/CSS frontend to index docs, ask questions with real-time sources, and launch evaluations.

**Tech Stack:** FastAPI, LangChain, ChromaDB, Ollama (qwen3:4b & nomic-embed-text), MLflow, HTML/CSS/JS.

---

### Task 1: Environment & Dependency Setup

**Objective:** Configure dependencies for the FastAPI, LangChain, ChromaDB, and MLflow backend.

**Files:**
- Modify: `backend/requirements.txt`

**Step 1: Write requirements file**

Update `backend/requirements.txt` with:
```text
fastapi==0.115.0
uvicorn==0.34.0
pydantic==2.10.0
python-multipart==0.0.18
pypdf==5.1.0
httpx==0.28.0
numpy==2.1.0
tiktoken==0.8.0
langchain==0.3.0
langchain-community==0.3.0
chromadb==0.5.15
mlflow==2.16.0
pandas==2.2.2
```

**Step 2: Install dependencies**

Run:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Expected: Successfully installed all packages without conflicts.

**Step 3: Verify installations**

Run:
```bash
python3 -c "import langchain, mlflow, chromadb; print('All clear!')"
```
Expected: `All clear!`

**Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: setup dependencies for RAG and MLflow"
```

---

### Task 2: Core LangChain RAG Engine

**Objective:** Write the document ingestion and retrieval engine using LangChain and ChromaDB.

**Files:**
- Create: `backend/rag_engine.py`
- Create: `tests/test_rag_engine.py`

**Step 1: Write tests for RAG Ingestion & Retrieval**

Create `tests/test_rag_engine.py`:
```python
import os
import shutil
import pytest
from backend.rag_engine import RAGEngine

@pytest.fixture
def temp_db_dir():
    db_dir = "./test_chroma_db"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    yield db_dir
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

def test_ingest_and_retrieve(temp_db_dir):
    engine = RAGEngine(persist_directory=temp_db_dir)
    
    # Test text
    test_text = "Alex is an MLOps and AI Engineer who built a multi-agent system with 6 workers."
    
    # Index text
    engine.index_text(test_text, chunk_size=50, chunk_overlap=10)
    
    # Query vector store
    results = engine.retrieve("How many workers did Alex build?", k=1)
    
    assert len(results) > 0
    assert "multi-agent system" in results[0].page_content
```

**Step 2: Write Minimal Implementation**

Create `backend/rag_engine.py`:
```python
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class RAGEngine:
    def __init__(self, persist_directory="./chroma_db", embedding_model="nomic-embed-text"):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434"
        )
        self.vector_store = None
        self._init_db()

    def _init_db(self):
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

    def index_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50, metadata: dict = None):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        metadatas = [metadata or {} for _ in chunks]
        
        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            metadatas=metadatas
        )

    def index_pdf(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        self.index_text(full_text, chunk_size, chunk_overlap, metadata={"source": os.path.basename(file_path)})

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)
```

**Step 3: Run the test to verify**

Run:
```bash
pytest tests/test_rag_engine.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add backend/rag_engine.py tests/test_rag_engine.py
git commit -m "feat: core RAG ingestion and retrieval with ChromaDB"
```

---

### Task 3: LLM Integration & Generation

**Objective:** Implement streaming generation using local Ollama model with prompt formatting.

**Files:**
- Create: `backend/llm_client.py`
- Create: `tests/test_llm_client.py`

**Step 1: Write test for LLM Client**

Create `tests/test_llm_client.py`:
```python
import pytest
from backend.llm_client import LLMClient

def test_generate_sync():
    client = LLMClient(model="qwen3:1.7b")
    response = client.generate("Say 'Hello' and nothing else.")
    assert "hello" in response.lower()
```

**Step 2: Write Implementation with Streaming & Prompt formatting**

Create `backend/llm_client.py`:
```python
import httpx
from typing import Generator

class LLMClient:
    def __init__(self, model="qwen3:4b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": {
                "temperature": temperature
            },
            "stream": False
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()

    def generate_stream(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.2) -> Generator[str, None, None]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": {
                "temperature": temperature
            },
            "stream": True
        }
        with httpx.stream("POST", url, json=payload, timeout=30.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    import json
                    try:
                        data = json.loads(line)
                        yield data.get("response", "")
                    except json.JSONDecodeError:
                        continue
```

**Step 3: Run the test**

Run:
```bash
pytest tests/test_llm_client.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add backend/llm_client.py tests/test_llm_client.py
git commit -m "feat: local LLM integration using Ollama"
```

---

### Task 4: Evaluation Judge & Golden Dataset

**Objective:** Set up the evaluation metrics and the Golden Dataset of questions, reference answers, and LLM-as-judge scoring.

**Files:**
- Create: `eval/golden_dataset.json`
- Create: `eval/evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write Golden Dataset**

Create `eval/golden_dataset.json` containing specific career/portfolio QA targets for Alex:
```json
[
  {
    "id": "q1",
    "question": "What is Alex's background and core stack?",
    "reference": "Alex is an MLOps and AI Engineer with 4 years of IT operations experience. His core stack includes Python, Docker, PostgreSQL, Prometheus, Grafana, and tools like LangChain and MLflow."
  },
  {
    "id": "q2",
    "question": "Describe Alex's flagship multi-agent system project.",
    "reference": "Alex built a flagship 6-agent AI system running entirely on local architecture (M1 Mac) and free services. It handled over 22,000 requests, has 52 test cases, and runs with $0 cloud spend."
  },
  {
    "id": "q3",
    "question": "What are Alex's trading execution rules?",
    "reference": "Alex trades MNQ1 only with 0.25 contracts and a stop loss of 200-334 ticks. He has a limit of 3 trades per day and stops after 2 losses, with a max daily loss of $500."
  }
]
```

**Step 2: Write tests for Evaluation Engine**

Create `tests/test_evaluator.py`:
```python
from eval.evaluator import RAGEvaluator

def test_eval_metrics():
    evaluator = RAGEvaluator(judge_model="qwen3:1.7b")
    
    # Perfect alignment test
    context = "Alex is an MLOps Engineer who built a 6-agent system on his M1 Mac."
    answer = "Alex built a 6-agent system on an M1 Mac as an MLOps Engineer."
    reference = "Alex is an MLOps Engineer who created a 6-agent system using local M1 Mac resources."
    
    faithfulness = evaluator.score_faithfulness(context, answer)
    relevance = evaluator.score_relevance("What did Alex build?", answer, reference)
    
    assert 1.0 <= faithfulness <= 5.0
    assert 1.0 <= relevance <= 5.0
```

**Step 3: Write Evaluator Implementation**

Create `eval/evaluator.py`:
```python
import json
import re
from backend.llm_client import LLMClient

class RAGEvaluator:
    def __init__(self, judge_model="qwen3:4b"):
        self.llm = LLMClient(model=judge_model)

    def score_faithfulness(self, context: str, answer: str) -> float:
        """Measure if the answer is grounded strictly in the provided context (Faithfulness)."""
        prompt = f"""
You are an expert evaluation judge. You are evaluating whether a generated answer is grounded in the provided context.
Context:
{context}

Answer:
{answer}

Rate the Faithfulness of the answer on a scale from 1 to 5:
1: The answer contains completely hallucinated facts not in the context.
3: The answer has some grounded facts but adds unverified assumptions.
5: The answer is 100% grounded and fully backed by the context.

Respond with ONLY a single digit score between 1 and 5. Do not write anything else.
Score:"""
        try:
            res = self.llm.generate(prompt, temperature=0.0)
            score = re.search(r"[1-5]", res)
            return float(score.group(0)) if score else 3.0
        except Exception:
            return 3.0

    def score_relevance(self, query: str, answer: str, reference: str) -> float:
        """Measure if the answer accurately responds to the query and matches the ground truth (Relevance)."""
        prompt = f"""
You are an expert evaluation judge. You are comparing a generated answer against a ground truth reference for a given query.
Query: {query}
Reference Ground Truth: {reference}
Generated Answer: {answer}

Rate the Relevance and Correctness of the answer on a scale from 1 to 5:
1: Completely irrelevant or wrong.
3: Partially correct but misses key information.
5: Excellent answer, highly accurate and directly answers the query.

Respond with ONLY a single digit score between 1 and 5. Do not write anything else.
Score:"""
        try:
            res = self.llm.generate(prompt, temperature=0.0)
            score = re.search(r"[1-5]", res)
            return float(score.group(0)) if score else 3.0
        except Exception:
            return 3.0

    def evaluate_retrieval_accuracy(self, retrieved_contexts: list, reference: str) -> float:
        """Measure what percentage of critical reference words are present in retrieved contexts."""
        ref_words = set(re.findall(r"\w+", reference.lower()))
        # Filter stop words to get meaningful keywords
        stop_words = {"is", "an", "and", "with", "the", "who", "his", "includes", "of", "in", "on", "a", "it", "has", "for", "only"}
        keywords = {w for w in ref_words if len(w) > 3 and w not in stop_words}
        
        if not keywords:
            return 1.0
            
        combined_context = " ".join([c.page_content.lower() for c in retrieved_contexts])
        matches = sum(1 for kw in keywords if kw in combined_context)
        return matches / len(keywords)
```

**Step 4: Run the test**

Run:
```bash
pytest tests/test_evaluator.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add eval/golden_dataset.json eval/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluation judge and golden dataset"
```

---

### Task 5: MLflow Experiment Tracking

**Objective:** Connect the RAG pipeline configurations and evaluation outcomes to MLflow.

**Files:**
- Create: `backend/tracker.py`
- Create: `tests/test_tracker.py`

**Step 1: Write test for Tracker**

Create `tests/test_tracker.py`:
```python
import mlflow
from backend.tracker import MLflowTracker

def test_tracker_logging():
    tracker = MLflowTracker(experiment_name="test-rag-experiment")
    
    with tracker.start_run(run_name="test_run") as run:
        tracker.log_params({
            "chunk_size": 500,
            "chunk_overlap": 50,
            "model": "qwen3:4b"
        })
        tracker.log_metrics({
            "avg_faithfulness": 4.5,
            "avg_relevance": 4.8,
            "retrieval_accuracy": 0.90
        })
    
    # Verify via mlflow client
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("test-rag-experiment")
    assert exp is not None
```

**Step 2: Write MLflow Tracker class**

Create `backend/tracker.py`:
```python
import mlflow

class MLflowTracker:
    def __init__(self, experiment_name="rag-eval-system", tracking_uri="http://localhost:5000"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Configure local tracking
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            mlflow.create_experiment(self.experiment_name)
        except Exception:
            # Experiment already exists
            pass
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_artifact(self, file_path: str):
        mlflow.log_artifact(file_path)
```

**Step 3: Run the test**

*Note: Ensure an MLflow local server or local filesystem directory is available. MLflow defaults to local file storage if the URI tracking server is offline.*

Run:
```bash
pytest tests/test_tracker.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add backend/tracker.py tests/test_tracker.py
git commit -m "feat: mlflow experiment tracking integration"
```

---

### Task 6: FastAPI Backend Endpoints

**Objective:** Write FastAPI routers and startup scripts to connect frontend commands to our RAG pipeline, local evaluations, and MLflow logging.

**Files:**
- Create: `backend/app.py`
- Create: `tests/test_app.py`

**Step 1: Write API tests**

Create `tests/test_app.py`:
```python
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
```

**Step 2: Write FastAPI Application Code**

Create `backend/app.py`:
```python
import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_engine import RAGEngine
from backend.llm_client import LLMClient
from backend.tracker import MLflowTracker
from eval.evaluator import RAGEvaluator

app = FastAPI(title="RAG Eval & Tracking System")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
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
        with open(temp_path, "wb") as f:
            shutil_content = await file.read()
            f.write(shutil_content)
        
        rag.index_pdf(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        os.remove(temp_path)
        return {"status": "success", "message": f"Indexed {file.filename}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    # Retrieve documents
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
        # First send the metadata/sources
        yield f"__METADATA__:{json.dumps(sources)}\n"
        for chunk in llm.generate_stream(prompt, system_prompt="Answer directly and truthfully."):
            yield chunk

    return StreamingResponse(stream_response(), media_type="text/event-stream")

@app.post("/evaluate")
def evaluate_endpoint(chunk_size: int = Form(500), chunk_overlap: int = Form(50)):
    # Run evaluation suite using Golden Dataset
    dataset_path = "../eval/golden_dataset.json"
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
            
            # Retrieve
            docs = rag.retrieve(q, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate answer
            prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
            generated_answer = llm.generate(prompt)
            
            # Evaluate
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

        # Log parameters & metrics to MLflow
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
```

**Step 3: Run FastAPI App tests**

Run:
```bash
pytest tests/test_app.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add backend/app.py tests/test_app.py
git commit -m "feat: fastapi backend router endpoints"
```

---

### Task 7: Frontend Web Interface

**Objective:** Create an interactive UI to run evaluations, index documents, and chat with streaming results and source visualization.

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/style.css`
- Create: `frontend/app.js`

**Step 1: HTML Structure**

Create `frontend/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Eval & MLflow Tracker</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>⚡ RAG Evaluation & MLflow Tracking System</h1>
            <p class="subtitle">Evolving Local AI Engineering & Performance Diagnostics</p>
        </header>

        <div class="grid">
            <!-- Left Panel: Settings and Upload -->
            <section class="card settings">
                <h2>📁 Ingestion Settings</h2>
                <div class="form-group">
                    <label for="chunk_size">Chunk Size</label>
                    <input type="number" id="chunk_size" value="500">
                </div>
                <div class="form-group">
                    <label for="chunk_overlap">Chunk Overlap</label>
                    <input type="number" id="chunk_overlap" value="50">
                </div>
                
                <hr>

                <h2>📄 Upload Document</h2>
                <form id="upload-form">
                    <input type="file" id="pdf-file" accept=".pdf" required>
                    <button type="submit" class="btn btn-primary">Index PDF</button>
                </form>
                <div id="upload-status"></div>

                <hr>

                <h2>📊 Evaluation Suite</h2>
                <p>Run full automated metrics evaluation across the Golden Dataset.</p>
                <button id="run-eval" class="btn btn-secondary">Run Eval Suite</button>
                <div id="eval-status"></div>
            </section>

            <!-- Right Panel: Q&A Chat -->
            <section class="card chat">
                <h2>💬 Document Q&A</h2>
                <div id="chat-messages" class="chat-box"></div>
                <form id="chat-form">
                    <input type="text" id="chat-input" placeholder="Ask about Alex's portfolio, stack, or trading rules..." required>
                    <button type="submit" class="btn btn-success">Send</button>
                </form>
            </section>
        </div>

        <!-- Evaluation Results Section -->
        <section class="card eval-results hidden" id="results-card">
            <h2>📈 Latest Evaluation Metrics (Logged to MLflow)</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <span class="metric-label">Faithfulness (LLM-as-judge)</span>
                    <span class="metric-value" id="val-faithfulness">-</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Answer Relevance (LLM-as-judge)</span>
                    <span class="metric-value" id="val-relevance">-</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Retrieval Accuracy (Keyword)</span>
                    <span class="metric-value" id="val-retrieval">-</span>
                </div>
            </div>
            <h3>Detailed Evaluation Run</h3>
            <div id="eval-table-container"></div>
        </section>
    </div>
    <script src="app.js"></script>
</body>
</html>
```

**Step 2: Simple Clean CSS**

Create `frontend/style.css`:
```css
:root {
    --bg-color: #121214;
    --card-bg: #1a1a1e;
    --text-color: #e1e1e6;
    --text-muted: #8d8d99;
    --primary: #00e676;
    --secondary: #2979ff;
    --success: #00e676;
    --border-color: #29292e;
}

body {
    background-color: var(--bg-color);
    color: var(--text-color);
    font-family: 'SF Pro Display', -apple-system, sans-serif;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

header {
    text-align: center;
    margin-bottom: 30px;
}

h1 {
    margin: 0;
    color: #fff;
}

.subtitle {
    color: var(--text-muted);
}

.grid {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 20px;
}

.card {
    background-color: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
}

.form-group {
    margin-bottom: 15px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

input[type="number"], input[type="text"], input[type="file"] {
    width: 100%;
    padding: 10px;
    background-color: #121214;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: #fff;
    box-sizing: border-box;
}

.btn {
    display: block;
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    margin-top: 10px;
    transition: background 0.2s;
}

.btn-primary { background-color: var(--primary); color: #000; }
.btn-secondary { background-color: var(--secondary); color: #fff; }
.btn-success { background-color: var(--success); color: #000; }

.chat-box {
    height: 400px;
    overflow-y: auto;
    border: 1px solid var(--border-color);
    padding: 15px;
    border-radius: 4px;
    background-color: #121214;
    margin-bottom: 15px;
}

.chat-message {
    margin-bottom: 15px;
}

.chat-message.user { text-align: right; color: var(--primary); }
.chat-message.assistant { text-align: left; }

.sources {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 5px;
    padding-left: 10px;
    border-left: 2px solid var(--border-color);
}

.hidden { display: none !important; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-bottom: 20px;
}

.metric-box {
    background-color: #121214;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 15px;
    text-align: center;
}

.metric-label {
    display: block;
    font-size: 14px;
    color: var(--text-muted);
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: var(--primary);
}
```

**Step 3: App Logic with Streaming EventSource**

Create `frontend/app.js`:
```javascript
const BACKEND_URL = "http://localhost:8000";

// Handle Chat Submission
document.getElementById("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("chat-input");
    const query = input.value;
    if (!query) return;

    appendMessage("user", query);
    input.value = "";

    const messageDiv = appendMessage("assistant", "Thinking...");
    
    const response = await fetch(`${BACKEND_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: query,
            chunk_size: parseInt(document.getElementById("chunk_size").value),
            chunk_overlap: parseInt(document.getElementById("chunk_overlap").value)
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantText = "";
    messageDiv.innerHTML = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        
        const lines = chunk.split("\n");
        for (const line of lines) {
            if (line.startsWith("__METADATA__:")) {
                const metadata = JSON.parse(line.replace("__METADATA__:", ""));
                renderSources(messageDiv, metadata);
            } else if (line) {
                assistantText += line;
                const textNode = document.createElement("span");
                textNode.innerText = line;
                messageDiv.insertBefore(textNode, messageDiv.querySelector(".sources"));
            }
        }
    }
});

// Handle PDF Indexing
document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const status = document.getElementById("upload-status");
    status.innerText = "Uploading & Indexing...";
    
    const formData = new FormData();
    formData.append("file", document.getElementById("pdf-file").files[0]);
    formData.append("chunk_size", document.getElementById("chunk_size").value);
    formData.append("chunk_overlap", document.getElementById("chunk_overlap").value);

    try {
        const res = await fetch(`${BACKEND_URL}/index-pdf`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        status.innerText = data.message || "Success!";
    } catch (err) {
        status.innerText = "Error indexing PDF.";
    }
});

// Handle Eval Execution
document.getElementById("run-eval").addEventListener("click", async () => {
    const status = document.getElementById("eval-status");
    status.innerText = "Running evaluation suite (please wait)...";

    const formData = new FormData();
    formData.append("chunk_size", document.getElementById("chunk_size").value);
    formData.append("chunk_overlap", document.getElementById("chunk_overlap").value);

    try {
        const res = await fetch(`${BACKEND_URL}/evaluate`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        status.innerText = "Evaluation Completed!";
        
        // Show metrics card
        document.getElementById("results-card").classList.remove("hidden");
        document.getElementById("val-faithfulness").innerText = data.metrics.faithfulness.toFixed(2);
        document.getElementById("val-relevance").innerText = data.metrics.relevance.toFixed(2);
        document.getElementById("val-retrieval").innerText = (data.metrics.retrieval_accuracy * 100).toFixed(0) + "%";
        
        renderEvalTable(data.details);
    } catch (err) {
        status.innerText = "Error running evaluation.";
    }
});

function appendMessage(sender, text) {
    const chatMessages = document.getElementById("chat-messages");
    const div = document.createElement("div");
    div.className = `chat-message ${sender}`;
    div.innerHTML = `<strong>${sender === 'user' ? 'You' : 'System'}:</strong> <span>${text}</span>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function renderSources(messageDiv, metadata) {
    const sourcesDiv = document.createElement("div");
    sourcesDiv.className = "sources";
    sourcesDiv.innerHTML = "<strong>Sources:</strong><br>";
    metadata.forEach((src, idx) => {
        const docName = src.metadata.source || "Indexed Doc";
        sourcesDiv.innerHTML += `[${idx+1}] ${docName}: "${src.content.substring(0, 100)}..."<br>`;
    });
    messageDiv.appendChild(sourcesDiv);
}

function renderEvalTable(details) {
    const container = document.getElementById("eval-table-container");
    let html = `<table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
        <thead>
            <tr style="border-bottom: 2px solid var(--border-color); text-align: left;">
                <th style="padding: 10px;">Question</th>
                <th style="padding: 10px;">Faithfulness</th>
                <th style="padding: 10px;">Relevance</th>
                <th style="padding: 10px;">Retrieval Acc</th>
            </tr>
        </thead>
        <tbody>`;
    details.forEach(item => {
        html += `<tr style="border-bottom: 1px solid var(--border-color);">
            <td style="padding: 10px;">${item.question}</td>
            <td style="padding: 10px;">${item.faithfulness}</td>
            <td style="padding: 10px;">${item.relevance}</td>
            <td style="padding: 10px;">${(item.retrieval_accuracy * 100).toFixed(0)}%</td>
        </tr>`;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;
}
```

**Step 4: Commit**

```bash
git add frontend/index.html frontend/style.css frontend/app.js
git commit -m "feat: simple frontend dashboard interface"
```

---

### Task 8: Verification & Cleanup

**Objective:** Verify that the system launches cleanly, docker services run, and we can trigger baseline evaluations logging directly to local MLflow backend.

**Files:**
- Create: `backend/run.py`

**Step 1: Write run orchestrator script**

Create `backend/run.py` to start MLflow server in parallel with FastAPI:
```python
import subprocess
import time
import sys

def main():
    print("Starting MLflow Tracking Server on port 5000...")
    mlflow_process = subprocess.Popen(
        ["mlflow", "server", "--host", "127.0.0.1", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./mlruns"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    print("Starting FastAPI Backend on port 8000...")
    try:
        subprocess.run(
            ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
            check=True
        )
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        mlflow_process.terminate()

if __name__ == "__main__":
    main()
```

**Step 2: Test core pipeline works cleanly**

Run tests:
```bash
pytest
```
Expected: All tests pass cleanly.

**Step 3: Commit**

```bash
git add backend/run.py
git commit -m "feat: run orchestrator script with MLflow server and FastAPI"
```
