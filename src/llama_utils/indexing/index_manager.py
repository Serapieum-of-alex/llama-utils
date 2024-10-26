from typing import List
from llama_index.core.indices.base import BaseIndex
from llama_index.core import StorageContext
from llama_index.core import load_indices_from_storage
from llama_utils.utils.config_loader import ConfigLoader

ConfigLoader()


class IndexManager:
    """A class to manage multiple indexes, handling updates, deletions, and retrieval operations."""

    def __init__(self, ids: List[str], indexes: List[BaseIndex]):
        self._indexes = indexes
        self._ids = ids

    @classmethod
    def create_from_storage(cls, storage_dir: str) -> "IndexManager":
        """Reads indexes from storage."""
        storage = StorageContext.from_defaults(persist_dir=storage_dir)
        index_instructs = storage.index_store.index_structs()
        index_ids = [index_i.index_id for index_i in index_instructs]
        indexes = load_indices_from_storage(storage)
        return cls(index_ids, indexes)

    @property
    def indexes(self) -> List[BaseIndex]:
        return self._indexes

    @indexes.setter
    def indexes(self, indexes: List[BaseIndex]):
        self._indexes = indexes

    @property
    def ids(self) -> List[str]:
        return self._ids
