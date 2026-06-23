"""Start RAG server and run test query."""
import os, sys, json, time, threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read key from Hermes .env using subprocess to avoid shell interpretation
import subprocess
result = subprocess.run(
    ['python3', '-c', """
with open('/Users/alex/.hermes/.env', 'r') as f:
    for line in f:
        if 'OPENROUTER_API_KEY=' in line and not line.strip().startswith('#'):
            key = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
            print(key)
            break
"""],
    capture_output=True, text=True
)
key = result.stdout.strip()

if not key or len(key) < 10:
    print(f"ERROR: Could not read key. stdout: {result.stdout[:50]} stderr: {result.stderr[:50]}")
    sys.exit(1)

print(f"Key: {key[:15]}... (len={len(key)})")
os.environ["OPENROUTER_API_KEY"] = key

from app_v2 import app
import uvicorn
import httpx

def run_server():
    uvicorn.run(app, host="localhost", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

with httpx.Client(timeout=30.0) as client:
    resp = client.post("http://localhost:8000/query", json={
        "query": "What is Alex trading strategy?",
        "k": 2,
        "use_expansion": False
    })
    print(f"Status: {resp.status_code}")
    raw = resp.text
    print(f"Response length: {len(raw)}")
    print(f"Raw: {raw[:500]}")
