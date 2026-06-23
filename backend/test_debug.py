"""Debug: test owl-alpha directly with the exact RAG prompt."""
import os, json, time, httpx

key = "sk-or-..."

# Test 1: Simple prompt (no citations)
url = "https://openrouter.ai/api/v1/chat/completions"
payload = {
    "model": "openrouter/owl-alpha",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one word."}
    ],
    "max_tokens": 20
}
with httpx.Client(timeout=15.0) as client:
    resp = client.post(url, json=payload, headers={
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/test"
    })
    print(f"Test 1 (simple): {resp.status_code}")
    if resp.status_code == 200:
        print(f"  Response: {resp.json()['choices'][0]['message']['content']}")

time.sleep(3)

# Test 2: Prompt with source citations
payload2 = {
    "model": "openrouter/owl-alpha",
    "messages": [
        {"role": "system", "content": "You are a precise RAG assistant. Every claim must be cited."},
        {"role": "user", "content": "Answer using ONLY the sources below.\n\n[1] (source: trading)\nAlex trades MNQ1 with a 73% win rate.\n\nQuestion: What is Alex's trading strategy?\n\nAnswer (with citations):"}
    ],
    "max_tokens": 200
}
with httpx.Client(timeout=15.0) as client:
    resp2 = client.post(url, json=payload2, headers={
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/test"
    })
    print(f"\nTest 2 (with citations): {resp2.status_code}")
    if resp2.status_code == 200:
        data = resp2.json()
        content = data['choices'][0]['message']['content']
        refusal = data['choices'][0]['message'].get('refusal', '')
        print(f"  Content ({len(content)} chars): {content[:200]}")
        print(f"  Refusal: {refusal}")
        print(f"  Full message: {json.dumps(data['choices'][0]['message'], indent=2)[:500]}")
    else:
        print(f"  Error: {resp2.text[:200]}")
