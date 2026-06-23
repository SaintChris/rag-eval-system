"""Debug: check if the key is loaded correctly."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simulate what the engine does
hermes_env = os.path.expanduser('~/.hermes/.env')
print(f"Hermes .env exists: {os.path.exists(hermes_env)}")

key = None
with open(hermes_env, 'r') as f:
    for line in f:
        line = line.strip()
        if 'OPENROUTER_API_KEY=*** in line and not line.startswith('#'):
            key = line.split('=', 1)[1].strip().strip('"').strip("'")
            break

if key:
    print(f"Key loaded: {key[:15]}... (len={len(key)})")
    print(f"Key valid: {len(key) > 10 and '***' not in key}")
else:
    print("ERROR: No key found!")
