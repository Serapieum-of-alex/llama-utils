from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext


class VectorStore:
    """A class to manage vector storage and retrieval."""

    def __init__(self, storage_backend: str = "simple"):
        # Initialize with the desired vector storage backend (e.g., Qdrant, FAISS)
        if storage_backend == "simple":
            self._store = self._create_simple_storage_context()

    @staticmethod
    def _create_simple_storage_context() -> StorageContext:
        return StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=SimpleVectorStore(),
            index_store=SimpleIndexStore(),
        )

    @property
    def store(self) -> StorageContext:
        return self._store

    def save_store(self, store_dir: str):
        self.store.persist(persist_dir=store_dir)

    def load_store(self, store_dir: str):
        self._store = StorageContext.from_defaults(persist_dir=store_dir)
