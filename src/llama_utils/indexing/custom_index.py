"""A custom class for creating indexes using Llama."""

from typing import List, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document


class CustomIndex:
    """A Custom class for creating indexes using Llama."""

    def __init__(self, index: VectorStoreIndex):
        """Initialize the CustomIndex object.

        Parameters
        ----------
        index: VectorStoreIndex
            The index object.
        """
        if not isinstance(index, VectorStoreIndex):
            raise ValueError("The index should be an instance of VectorStoreIndex")
        self._index = index

    @property
    def index(self) -> VectorStoreIndex:
        """Return the index object."""
        return self._index

    @classmethod
    def create_from_documents(cls, document: List[Union[Document, str]]):
        """Create a new index from a document.

        Parameters
        ----------
        document: List[Document]
            The document to create the index from.

        Returns
        -------
        IndexManager
            The new index manager object

        Examples
        --------
        >>> doc = Document(text="text")
        >>> index = CustomIndex.create_from_documents([doc]) # doctest: +SKIP
        >>> type(index) # doctest: +SKIP
        <class 'llama_utils.indexing.custom_index.CustomIndex'>
        """
        docs = [Document(text=doc) if isinstance(doc, str) else doc for doc in document]
        index = VectorStoreIndex.from_documents(docs)
        return cls(index)
