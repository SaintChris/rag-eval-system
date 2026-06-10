import os
import shutil
import pytest
from unittest.mock import patch, MagicMock
from backend.rag_engine import RAGEngine


@pytest.fixture
def temp_db_dir():
    db_dir = "./test_chroma_db"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    yield db_dir
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)


def test_ingest_and_retrieve(temp_db_dir):
    # Mock SentenceTransformer to avoid slow model loading on CPU
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])

    with patch("rag_engine.SentenceTransformer", return_value=mock_model):
        engine = RAGEngine(persist_directory=temp_db_dir)
        test_text = "Alex is an MLOps and AI Engineer who built a multi-agent system with 6 workers."
        engine.index_text(test_text, chunk_size=50, chunk_overlap=10)
        results = engine.retrieve("How many workers did Alex build?", k=1)
        assert len(results) > 0
