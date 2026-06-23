"""Debug: trace exactly what happens during key loading."""
import os
import sys

# Simulate server startup
print("=== Simulating server key loading ===")
print(f"CWD: {os.getcwd()}")
print(f"__file__: {os.path.abspath(__file__)}")

# This is what _load_openrouter_key does
hermes_env = os.path.expanduser("~/.hermes/.env")
print(f"\nHermes .env path: {hermes_env}")
print(f"Hermes .env exists: {os.path.exists(hermes_env)}")

if os.path.exists(hermes_env):
    with open(hermes_env, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if 'OPENROUTER_API_KEY=*** in line and not line.startswith('#'):
                key = line.split('=', 1)[1].strip().strip('"').strip("'")
                print(f"\nLine {i}: OPENROUTER_API_KEY found")
                print(f"  Raw value: {repr(key)}")
                print(f"  Length: {len(key)}")
                print(f"  Valid: {len(key) > 10 and '***' not in key}")
                break

# Also check local .env
local_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
print(f"\nLocal .env path: {local_env}")
print(f"Local .env exists: {os.path.exists(local_env)}")

if os.path.exists(local_env):
    with open(local_env, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if 'OPENROUTER_API_KEY=*** in line and not line.startswith('#'):
                key = line.split('=', 1)[1].strip().strip('"').strip("'")
                print(f"\nLine {i}: OPENROUTER_API_KEY found")
                print(f"  Raw value: {repr(key)}")
                print(f"  Length: {len(key)}")
                print(f"  Valid: {len(key) > 10 and '***' not in key}")
                break

# Test direct API call with the key
print("\n=== Testing API call ===")
import httpx
hermes_env2 = os.path.expanduser("~/.hermes/.env")
key = None
with open(hermes_env2, 'r') as f:
    for line in f:
        if 'OPENROUTER_API_KEY=*** in line and not line.strip().startswith('#'):
            key = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
            break

if key:
    print(f"Using key: {key[:15]}... (len={len(key)})")
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "openrouter/owl-alpha",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 10
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, json=payload, headers={
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://github.com/test"
        })
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: {resp.json()['choices'][0]['message']['content']}")
        else:
            print(f"Error: {resp.text[:200]}")
