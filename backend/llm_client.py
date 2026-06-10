import json
import httpx
from typing import Generator


class LLMClient:
    def __init__(self, model="qwen3:4b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": {"temperature": temperature},
            "stream": False
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()

    def generate_stream(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.2) -> Generator[str, None, None]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": {"temperature": temperature},
            "stream": True
        }
        with httpx.stream("POST", url, json=payload, timeout=60.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get("response", "")
                    except json.JSONDecodeError:
                        continue
