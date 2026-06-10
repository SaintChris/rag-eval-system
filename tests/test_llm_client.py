import pytest
from unittest.mock import patch, MagicMock
from backend.llm_client import LLMClient


def test_generate_sync():
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Hello"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_response):
        client = LLMClient(model="qwen3:1.7b")
        response = client.generate("Say 'Hello' and nothing else.")
        assert "hello" in response.lower()


def test_generate_stream():
    mock_line = b'{"response": "Hello"}'

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_lines.return_value = [mock_line]

    with patch("httpx.stream", return_value=MagicMock(__enter__=MagicMock(return_value=mock_response), __exit__=MagicMock())):
        client = LLMClient(model="qwen3:1.7b")
        chunks = list(client.generate_stream("Say hello"))
        assert len(chunks) > 0
