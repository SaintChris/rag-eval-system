import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


class RAGEngine:
    def __init__(self, persist_directory="./chroma_db", embedding_model="nomic-embed-text"):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434"
        )
        self.vector_store = None
        self._init_db()

    def _init_db(self):
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

    def index_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50, metadata: dict = None):
        from langchain_core.documents import Document
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
