"""Start RAG server with the real key and latest code."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read key from Hermes .env
hermes_env = os.path.expanduser("~/.hermes/.env")
key = None
with open(hermes_env, 'r') as f:
    for line in f:
        line = line.strip()
        if 'OPENROUTER_API_KEY=' in line and not line.startswith('#'):
            key = line.split('=', 1)[1].strip().strip('"').strip("'")
            break

if not key or len(key) < 10 or '***' in key:
    print("ERROR: Could not read key from ~/.hermes/.env")
    print("Make sure OPENROUTER_API_KEY is set in ~/.hermes/.env")
    sys.exit(1)

print(f"Key: {key[:15]}... (len={len(key)})")
os.environ["OPENROUTER_API_KEY"] = key

from app_v2 import app
import uvicorn

print("Starting RAG v2 server on http://localhost:8000")
print("Press Ctrl+C to stop")
uvicorn.run(app, host="localhost", port=8000, log_level="info")
