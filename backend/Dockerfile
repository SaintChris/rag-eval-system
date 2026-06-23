# Slim Dockerfile for RAG v2
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY backend/ ./backend/
COPY eval/ ./eval/

RUN mkdir -p /app/chroma_db /app/logs

EXPOSE 8000

CMD ["uvicorn", "backend.app_v2:app", "--host", "[IP_ADDRESS]", "--port", "8000"]
