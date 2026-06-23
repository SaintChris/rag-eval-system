"""Read the OpenRouter key from Hermes .env file."""
with open('/Users/alex/.hermes/.env', 'r') as f:
    for line in f:
        if 'OPENROUTER_API_KEY=' in line and not line.strip().startswith('#'):
            key = line.strip().split('=', 1)[1].strip()
            print(f'Key length: {len(key)}')
            print(f'Key: {key}')
            break
