"""Start RAG server and run test query with debug logging."""
import os, sys, json, time, threading, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

result = subprocess.run(
    ['python3', '-c', """
with open('/Users/alex/.hermes/.env', 'r') as f:
    for line in f:
        if 'OPENROUTER_API_KEY=*** in line and not line.strip().startswith('#'):
            key = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
            print(key)
            break
"""],
    capture_output=True, text=True
)
key = result.stdout.strip()
os.environ["OPENROUTER_API_KEY"] = key

from app_v2 import app
import uvicorn
import httpx

def run_server():
    uvicorn.run(app, host="localhost", port=8000, log_level="info")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

with httpx.Client(timeout=30.0) as client:
    resp = client.post("http://localhost:8000/query", json={
        "query": "What is Alex trading strategy?",
        "k": 2,
        "use_expansion": False
    })
    raw = resp.text
    print(f"Status: {resp.status_code}")
    print(f"Response length: {len(raw)}")
    
    # Parse response
    if "__METADATA__" in raw:
        parts = raw.split("__METADATA__:")
        meta = json.loads(parts[1].split("\n")[0])
        answer = parts[1].split("\n", 1)[1] if "\n" in parts[1] else ""
        print(f"Docs: {len(meta)}")
        print(f"Answer length: {len(answer)}")
        print(f"Answer: {answer[:300]}")
    else:
        print(f"Raw: {raw[:300]}")
