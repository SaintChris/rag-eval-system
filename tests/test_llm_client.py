import pytest
from backend.llm_client import LLMClient


def test_generate_sync():
    client = LLMClient(model="qwen3:1.7b")
    response = client.generate("Say 'Hello' and nothing else.")
    assert "hello" in response.lower()
