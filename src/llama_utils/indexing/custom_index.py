"""A custom class for creating indexes using Llama."""

from typing import Dict, List, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from llama_utils.utils.helper_functions import generate_content_hash


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
        self._id = index.index_id
        self._index = index

    def __str__(self):
        """String representation of the CustomIndex object."""
        return f"""
        Index ID: {self.id}
        Number of Document: {len(self.doc_ids)}
        """

    @property
    def index(self) -> VectorStoreIndex:
        """Return the index object."""
        return self._index

    @property
    def vector_store(self) -> BasePydanticVectorStore:
        """Return the vector store."""
        return self.index.vector_store

    @property
    def doc_ids(self) -> List[str]:
        """Return the document IDs."""
        return list(self.index.ref_doc_info.keys())

    @property
    def id(self) -> str:
        """Return the index ID."""
        return self._id

    @property
    def embeddings(self) -> Dict[str, List[float]]:
        """Return the embeddings."""
        # the ref_ids is a mapping of text_id to ref_doc_id
        ref_ids = self.vector_store.data.text_id_to_ref_doc_id
        embedding_docs = self.vector_store.data.embedding_dict
        embeddings = {ref_ids[doc_id]: embedding_docs[doc_id] for doc_id in ref_ids}
        return embeddings

    @classmethod
    def create_from_documents(
        cls, document: List[Union[Document, str]], generate_id: bool = True
    ) -> "CustomIndex":
        """Create a new index from a document.

        Parameters
        ----------
        document: List[Document]
            The document to create the index from.
        generate_id: bool, optional, default is False.
            True if you want to generate a sha256 hash number as a doc_id based on the content of the nodes.

        Returns
        -------
        CustomIndex
            The new CustomIndex object

        Examples
        --------
        >>> doc = Document(text="text")
        >>> index = CustomIndex.create_from_documents([doc]) # doctest: +SKIP
        >>> type(index) # doctest: +SKIP
        <class 'llama_utils.indexing.custom_index.CustomIndex'>
        """
        docs = [Document(text=doc) if isinstance(doc, str) else doc for doc in document]
        # change the node.id to the content hash
        if generate_id:
            for doc in docs:
                doc.doc_id = generate_content_hash(doc.text)

        index = VectorStoreIndex.from_documents(docs)
        return cls(index)

    @classmethod
    def create_from_nodes(cls, nodes: List[TextNode]):
        """Create a new index from a node.

        Parameters
        ----------
        nodes: List[TextNode]
            The nodes to create the index from.

        Returns
        -------
        CustomIndex
            The new CustomIndex object

        Examples
        --------
        To create a new index you have to define the embedding model
        >>> from llama_utils.utils.config_loader import ConfigLoader
        >>> configs = ConfigLoader()
        >>> text_node = TextNode(text="text")
        >>> index = CustomIndex.create_from_nodes([text_node])
        >>> print(index) # doctest: +SKIP
        <BLANKLINE>
                Index ID: 8d57e294-fd17-43c9-9dec-a12aa7ea0751
                Number of Document: 0
        <BLANKLINE>
        As you see the added node is not a document, so the number of documents is 0.
        """
        index = VectorStoreIndex(nodes)
        return cls(index)
