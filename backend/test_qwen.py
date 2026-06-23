"""Test qwen3-coder:free directly."""
import os, sys, json, time, httpx

hermes_env = os.path.expanduser("~/.hermes/.env")
key = None
with open(hermes_env, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

print(f"Key length: {len(key)}")

url = "https://openrouter.ai/api/v1/chat/completions"
payload = {
    "model": "qwen/qwen3-coder:free",
    "messages": [
        {"role": "system", "content": "You are a precise RAG assistant. Every claim must be cited with [1], [2], etc."},
        {"role": "user", "content": "Answer using ONLY the sources below.\n\n[1] (source: trading)\nAlex's trading approach uses liquidity-based market structure analysis (SMC + ICT influence). He trades MNQ1 with a 73% win rate and +$10,352 profit.\n\nQuestion: What is Alex's trading strategy?\n\nAnswer (with citations):"}
    ],
    "temperature": 0.2,
    "max_tokens": 1024
}
with httpx.Client(timeout=30.0) as client:
    resp = client.post(url, json=payload, headers={
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/test"
    })
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        content = data['choices'][0]['message']['content']
        print(f"Response ({len(content)} chars):")
        print(content[:500])
    else:
        print(f"Error: {resp.text[:300]}")
