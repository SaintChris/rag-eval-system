"""Single model test with longer timeout."""
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
payload = {
    "model": "openrouter/owl-alpha",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 10
}
with httpx.Client(timeout=20.0) as client:
    resp = client.post(url, json=payload, headers={
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/test"
    })
    print(f"owl-alpha: {resp.status_code}")
    if resp.status_code == 200:
        print(f"  {resp.json()['choices'][0]['message']['content']}")
