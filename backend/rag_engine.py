"""
RAG Engine using SentenceTransformers for local embeddings.
Bypasses Ollama's slow /api/embeddings endpoint.
"""
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Wraps a sentence-transformers model toLangChain's Embeddings interface."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class RAGEngine:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = LocalEmbeddings()
        self.vector_store = None
        self._init_db()

    def _init_db(self):
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

    def index_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50, metadata: dict = None):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=c, metadata=metadata or {}) for c in chunks]
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    def index_pdf(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        self.index_text(full_text, chunk_size, chunk_overlap, metadata={"source": os.path.basename(file_path)})

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)
