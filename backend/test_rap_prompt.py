"""Test owl-alpha with exact RAG prompt format."""
import os, json, time, httpx

hermes_env = os.path.expanduser("~/.hermes/.env")
key = None
with open(hermes_env, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

url = "https://openrouter.ai/api/v1/chat/completions"

# Test 1: Simple prompt (works)
print("Test 1: Simple prompt")
payload1 = {
    "model": "openrouter/owl-alpha",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 10
}
with httpx.Client(timeout=15.0) as client:
    resp = client.post(url, json=payload1, headers={"Authorization": "Bearer " + key})
    print("  Status:", resp.status_code, "Response:", resp.json()["choices"][0]["message"]["content"][:50])

time.sleep(3)

# Test 2: RAG prompt with [1] citation
print("\nTest 2: RAG prompt with [1] citation")
rag_prompt = (
    "Answer using ONLY the sources below.\n\n"
    "[1] (source: trading)\n"
    "Alex's trading approach uses liquidity-based market structure analysis.\n\n"
    "Question: What is Alex's trading strategy?\n\n"
    "Answer (with citations):"
)
payload2 = {
    "model": "openrouter/owl-alpha",
    "messages": [
        {"role": "system", "content": "You are a precise RAG assistant. Every claim must be cited with [1], [2], etc."},
        {"role": "user", "content": rag_prompt}
    ],
    "max_tokens": 512
}
with httpx.Client(timeout=15.0) as client:
    resp = client.post(url, json=payload2, headers={"Authorization": "Bearer " + key})
    print("  Status:", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        refusal = data["choices"][0]["message"].get("refusal", "")
        print("  Content (" + str(len(content)) + " chars):", content[:200])
        print("  Refusal:", refusal)
        print("  Full:", json.dumps(data["choices"][0]["message"])[:300])
