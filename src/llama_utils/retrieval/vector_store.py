"""A module for managing vector storage and retrieval."""

from typing import Sequence, Union
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import Document, TextNode


class VectorStore:
    """A class to manage vector storage and retrieval."""

    def __init__(self, storage_backend: str = None):
        """Initialize the VectorStore.

        Parameters
        ----------
        storage_backend: str, optional, default=None
            The desired vector storage backend (e.g., Qdrant, FAISS). If none is provided, a simple storage context
            will be created.
        """
        # Initialize with the desired vector storage backend (e.g., Qdrant, FAISS)
        if storage_backend is None:
            self._store = self._create_simple_storage_context()

    @staticmethod
    def _create_simple_storage_context() -> StorageContext:
        """Create a simple storage context."""
        return StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=SimpleVectorStore(),
            index_store=SimpleIndexStore(),
        )

    @property
    def store(self) -> StorageContext:
        """Get the storage context."""
        return self._store

    def save_store(self, store_dir: str):
        """Save the store to a directory.

        Parameters
        ----------
        store_dir: str
            The directory to save the store.

        Returns
        -------
        None
        """
        self.store.persist(persist_dir=store_dir)

    def load_store(self, store_dir: str):
        """Load the store from a directory.

        Parameters
        ----------
        store_dir: str
            The directory containing the store.

        Returns
        -------
        None
        """
        self._store = StorageContext.from_defaults(persist_dir=store_dir)

    def add_docs(self, docs: Sequence[Union[Document, TextNode]]):
        """Add node to the store.

        Parameters
        ----------
        docs: Sequence[TextNode/Document]
            The node/documents to add to the store.

        Returns
        -------
        None
        """
        self.store.docstore.add_documents(docs)
