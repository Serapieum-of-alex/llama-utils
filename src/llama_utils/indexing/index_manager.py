from typing import List
from llama_index.core.indices.base import BaseIndex
from llama_index.core import StorageContext
from llama_index.core import load_indices_from_storage


class IndexManager:
    """A class to manage multiple indexes, handling updates, deletions, and retrieval operations."""

    def __init__(self, indexes: List[BaseIndex]):
        self._indexes = indexes

    @classmethod
    def create_from_storage(cls, storage_dir: str) -> List[BaseIndex]:
        """Reads indexes from storage."""
        storage = StorageContext.from_defaults(persist_dir=storage_dir)
        indexes = load_indices_from_storage(storage)
        return cls(indexes)

    @property
    def indexes(self) -> List[BaseIndex]:
        return self._indexes

    @indexes.setter
    def indexes(self, indexes: List[BaseIndex]):
        self._indexes = indexes
