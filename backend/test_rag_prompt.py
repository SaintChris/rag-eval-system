import os, time, httpx, json

hermes_env = os.path.expanduser('~/.hermes/.env')
key = None
with open(hermes_env) as f:
    for line in f:
        line = line.strip()
        if 'OPENROUTER_API_KEY=' in line and not line.startswith('#'):
            key = line.split('=', 1)[1].strip().strip('"').strip("'")
            break

time.sleep(2)

context = "[1] (source: trading)\nAlex's trading approach uses liquidity-based market structure analysis (SMC + ICT influence). His execution model: Sweep, Displacement, Retest, Continuation. He trades MNQ1 (NASDAQ-100 futures) with a 73% win rate and +$10,352 profit. His rules: Stop loss 200-334 ticks (never under 150). Target: minimum 1.5:1 to 2:1 risk-reward ratio. Maximum 3 trades per day. Stops trading after 2 consecutive losses. Maximum daily loss: $500. Position holding time: 1-2 hours. Primary trading window: 11:00 AM ET (shorts). Secondary windows: 6:00 PM and 11:00 PM ET (longs). He is banned from ES/MES after -$5,318 loss."

prompt = "Answer using ONLY the sources below. If the answer cannot be found, say \"I don't know.\"\n\nRules:\n- Every claim MUST be cited with [1], [2], etc.\n- Be direct and concise.\n- Do not make up information.\n\n## Sources:\n" + context + "\n\n## Question:\nWhat is Alex trading strategy?\n\n## Answer (with citations):\n"

url = 'https://openrouter.ai/api/v1/chat/completions'
payload = {
    'model': 'openrouter/owl-alpha',
    'messages': [
        {'role': 'system', 'content': 'You are a precise RAG assistant. Every claim must be cited with [1], [2], etc.'},
        {'role': 'user', 'content': prompt}
    ],
    'temperature': 0.2,
    'max_tokens': 1024
}
with httpx.Client(timeout=30.0) as client:
    resp = client.post(url, json=payload, headers={
        'Authorization': 'Bearer ' + key,
        'HTTP-Referer': 'https://github.com/SaintChris/rag-eval-system'
    })
    print('Status:', resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        content = data['choices'][0]['message']['content']
        print('Response (' + str(len(content)) + ' chars):')
        print(content[:500])
    else:
        print('Error:', resp.text[:300])
