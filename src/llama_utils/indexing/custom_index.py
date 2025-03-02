"""A custom class for creating indexes using Llama."""

from typing import Dict, List, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from llama_utils.utils.helper_functions import generate_content_hash
from llama_utils.utils.models import get_hugging_face_embedding


class CustomIndex:
    """A Custom class for creating indexes using Llama."""

    def __init__(self, index: VectorStoreIndex, embedding_model=None):
        """Initialize the CustomIndex object.

        Parameters
        ----------
        index: VectorStoreIndex
            The index object.
        embedding_model: Optional, default is None.
            The embedding model to use to create the embeddings in the index.

        """
        if not isinstance(index, VectorStoreIndex):
            raise ValueError("The index should be an instance of VectorStoreIndex")
        self._id = index.index_id
        self._index = index
        self._embedding_model = embedding_model or get_hugging_face_embedding()

    def __str__(self):
        """String representation of the CustomIndex object."""
        return f"""
        Index ID: {self.id}
        Number of Document: {len(self.doc_ids)}
        """

    def __repr__(self):
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
    def metadata(self) -> IndexDict:
        """metadata.

        Returns
        -------
        IndexDict
            The metadata of the index.

        Examples
        --------
        ```python
        >>> from llama_utils.utils.config_loader import ConfigLoader
        >>> config_loader = ConfigLoader()
        >>> text_node = TextNode(text="text")
        >>> index = CustomIndex.create_from_nodes([text_node])
        >>> metadata = index.metadata
        >>> type(metadata)
        <class 'llama_index.core.data_structs.data_structs.IndexDict'>
        >>> print(metadata) # doctest: +SKIP
        IndexDict(
            index_id='f543efc6-2d2c-451c-bfb6-ce7e1f0c3a51',
            summary=None,
            nodes_dict={'5bf5b272-3d2d-4f7c-9d12-be72de8646e1': '5bf5b272-3d2d-4f7c-9d12-be72de8646e1'},
            doc_id_dict={},
            embeddings_dict={}
        )
        ```
        """
        return self.index.index_struct

    @property
    def vector_store(self) -> BasePydanticVectorStore:
        """The vector store."""
        return self.index.vector_store

    @property
    def doc_ids(self) -> List[str]:
        """The document IDs. (only documents not the nodes)."""
        return list(self.index.ref_doc_info.keys())

    @property
    def id(self) -> str:
        """The index ID."""
        return self._id

    @property
    def embedding_model(self):
        """The embedding model."""
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, model):
        self._embedding_model = model

    @property
    def node_id_list(self) -> List[str]:
        """The node IDs."""
        return list(self.metadata.nodes_dict.keys())

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
        ```python
        >>> doc = Document(text="text")
        >>> index = CustomIndex.create_from_documents([doc]) # doctest: +SKIP
        >>> type(index) # doctest: +SKIP
        <class 'llama_utils.indexing.custom_index.CustomIndex'>
        ```
        """
        docs = [Document(text=doc) if isinstance(doc, str) else doc for doc in document]
        # change the node.id to the content hash
        if generate_id:
            for doc in docs:
                doc.node_id = generate_content_hash(doc.text)

        index = VectorStoreIndex.from_documents(docs)
        return cls(index)

    @classmethod
    def create_from_nodes(cls, nodes: List[TextNode]) -> "CustomIndex":
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
        ```python
        >>> from llama_utils.utils.config_loader import ConfigLoader
        >>> configs = ConfigLoader()
        >>> text_node = TextNode(text="text")
        >>> index = CustomIndex.create_from_nodes([text_node])
        >>> print(index) # doctest: +SKIP
        <BLANKLINE>
                Index ID: 8d57e294-fd17-43c9-9dec-a12aa7ea0751
                Number of Document: 0
        <BLANKLINE>
        ```
        As you see the added node is not a document, so the number of documents is 0.
        """
        index = VectorStoreIndex(nodes)
        return cls(index)

    def add_documents(self, documents: List[Document], generate_id: bool = True):
        """Add documents to the index.

        Parameters
        ----------
        documents: List[Document]
            The documents to add to the index.
        generate_id: bool, optional, default is False.
            True if you want to generate a sha256 hash number as a doc_id based on the content of the nodes.

        Raises
        ------
        ValueError
            If the documents are not a list of Document objects.
        ValueError
            If the documents are not instances of Document.

        Examples
        --------
        Set the ConfigLoader to define the embedding model that you want to use to create the embeddings in the index:
        ```python
        >>> from llama_utils.utils.config_loader import ConfigLoader
        >>> configs = ConfigLoader()

        ```
        Create a new index from a document:
        ```python
        >>> doc = Document(text="text", id_="doc 1")
        >>> index = CustomIndex.create_from_documents([doc])
        >>> print(index) # doctest: +SKIP
        <BLANKLINE>
                Index ID: 91dd8a18-3ab5-41ca-b8de-998077b9235c
                Number of Document: 1
        <BLANKLINE>
        ```
        Add a new document to the index:
        ```python
        >>> doc2 = Document(text="text2", id_="doc 2")

        ```
        The `add_documents` method has the `genereate_id` parameter, which is set to True by default to generate a
        sha256 hash number as a doc_id based on the content of the nodes:
        ```python
        >>> index.add_documents([doc2])
        >>> print(index.doc_ids) # doctest: +SKIP
        ['982d9e3eb996f559e633f4d194def3761d909f5a3b647d1a851fead67c32c9d1', 'fd848ca35a6281600b5da598c7cb4d5df561e0ee63ee7cec0e98e6049996f3ff']
        ```
        If you want to keep the same doc_id, you can set the `generate_id` parameter to False:
        ```python
        >>> index.add_documents([doc2], generate_id=False)
        >>> print(index.doc_ids) # doctest: +SKIP
        ['982d9e3eb996f559e633f4d194def3761d909f5a3b647d1a851fead67c32c9d1', 'doc 2']
        ```
        """
        if not isinstance(documents, list):
            raise ValueError("The documents should be a list of Document/TextNodes")

        for document in documents:
            if generate_id:
                document.node_id = generate_content_hash(document.text)

            if isinstance(document, Document):
                self.index.insert(document)
            elif isinstance(document, TextNode):
                document.embedding = self.embedding_model.get_text_embedding(
                    document.text
                )
                self.index.insert_nodes([document])
            else:
                raise ValueError(
                    "The document should be an instance of Document or TextNode"
                )
