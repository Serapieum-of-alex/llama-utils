from typing import List
from llama_index.core.indices.base import BaseIndex
from llama_index.core import load_indices_from_storage, VectorStoreIndex
from llama_utils.retrieval.storage import Storage


class IndexManager:
    """A class to manage multiple indexes, handling updates, deletions, and retrieval operations."""

    def __init__(self, ids: List[str], indexes: List[BaseIndex]):
        self._indexes = indexes
        self._ids = ids

    def __str__(self):
        return f"""
            ids={self.ids},
            indexes={self.indexes})
        """

    @classmethod
    def load_from_storage(cls, storage: Storage) -> "IndexManager":
        """Reads indexes from storage.

        Parameters
        ----------
        storage : Storage
            The storage object to read the indexes from.

        Returns
        -------
        IndexManager
            The index manager object

        Examples
        --------
        First we need to load the `ConfigLoader` to define the embedding model that was used to create the embeddings
        in the index.

            >>> from llama_utils.utils.config_loader import ConfigLoader
            >>> config_loader = ConfigLoader()

        Next, we load the storage object and the index manager object.

            >>> storage_dir = "examples/paul-graham-essay-storage"
            >>> storage_context = Storage.load(storage_dir)
            >>> index_manager = IndexManager.load_from_storage(storage_context)
            >>> print(index_manager) # doctest: +SKIP
            <BLANKLINE>
                ids=['8d57e294-fd17-43c9-9dec-a12aa7ea0751', 'edd0d507-9100-4cfb-8002-2267449c6668'],
                indexes=[
                    <llama_index.core.indices.vector_store_index.VectorStoreIndex object at 0x7f9f2a1e9d90>,
                    <llama_index.core.indices.vector_store_index.VectorStoreIndex object at 0x7f9f2a1e9e50>
                ])
            <BLANKLINE>
        """
        storage = storage.store
        index_instructs = storage.index_store.index_structs()
        index_ids = [index_i.index_id for index_i in index_instructs]
        indexes = load_indices_from_storage(storage)
        return cls(index_ids, indexes)

    @property
    def indexes(self) -> List[BaseIndex]:
        """Indexes"""
        return self._indexes

    @indexes.setter
    def indexes(self, indexes: List[BaseIndex]):
        self._indexes = indexes

    @property
    def ids(self) -> List[str]:
        """Index IDs"""
        return self._ids

    @classmethod
    def create_from_storage(cls, storage: Storage) -> "IndexManager":
        """Creates a new index.

        Parameters
        ----------
        storage : Storage
            The storage object to create the index from.

        Returns
        -------
        IndexManager
            The new index manager object
        """
        docstore = storage.docstore
        index = VectorStoreIndex(
            list(docstore.docs.values()), storage_context=storage.store
        )
        return cls([index.index_id], [index])
