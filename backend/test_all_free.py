"""Test multiple free models to find one that works right now."""
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
prompt_text = "Answer using ONLY the sources below.\n\n[1] (source: trading)\nAlex's trading approach uses liquidity-based market structure analysis (SMC + ICT influence). He trades MNQ1 with a 73% win rate and +$10,352 profit.\n\nQuestion: What is Alex's trading strategy?\n\nAnswer (with citations):"

models = [
    "openrouter/owl-alpha",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

for model in models:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise RAG assistant. Every claim must be cited with [1], [2], etc."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, json=payload, headers={
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://github.com/test"
        })
        if resp.status_code == 200:
            data = resp.json()
            content = data['choices'][0]['message']['content']
            print(f"OK  {model}")
            print(f"    ({len(content)} chars): {content[:200]}")
        elif resp.status_code == 429:
            print(f"RATE {model}")
        elif resp.status_code == 401:
            print(f"AUTH {model}")
        else:
            print(f"ERR  {model}: {resp.status_code} - {resp.text[:100]}")
    time.sleep(3)
