import os, time, httpx, json

hermes_env = os.path.expanduser('~/.hermes/.env')
key = None
with open(hermes_env) as f:
    for line in f:
        line = line.strip()
        if 'OPENROUTER_API_KEY=' in line and not line.startswith('#'):
            key = line.split('=', 1)[1].strip().strip('"').strip("'")
            break

print(f'Key length: {len(key)}')
print(f'Key prefix: {key[:15]}...' if key else 'NO KEY FOUND')

if key:
    time.sleep(3)
    url = 'https://openrouter.ai/api/v1/chat/completions'
    payload = {
        'model': 'openrouter/owl-alpha',
        'messages': [{'role': 'user', 'content': 'Say hello in one word.'}],
        'max_tokens': 20
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers={
            'Authorization': f'Bearer {key}',
            'HTTP-Referer': 'https://github.com/test'
        })
        print(f'Status: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            print(f'Response: {data["choices"][0]["message"]["content"]}')
        else:
            print(f'Error: {resp.text[:200]}')
