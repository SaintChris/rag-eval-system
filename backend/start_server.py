"""
Start RAG server with explicit API key.
Usage: OPENROUTER_API_KEY=*** python3 start_server.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Verify key is set
key = os.environ.get("OPENROUTER_API_KEY", "")
if not key or "***" in key:
    print("ERROR: Set OPENROUTER_API_KEY environment variable with the real key")
    sys.exit(1)

print(f"Key: {key[:10]}... (len={len(key)})")

from app_v2 import app
import uvicorn

print("Starting RAG v2 server on http://localhost:8000")
uvicorn.run(app, host="localhost", port=8000)
