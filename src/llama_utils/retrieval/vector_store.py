"""A module for managing vector storage and retrieval."""

import os
from typing import Sequence, Union, List, Dict
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import Document, TextNode
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
    SummaryExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_utils.config import Config
from llama_utils.utils.helper_functions import generate_content_hash

Config()

EXTRACTORS = dict(
    text_splitter=TokenTextSplitter,
    title=TitleExtractor,
    question_answer=QuestionsAnsweredExtractor,
    summary=SummaryExtractor,
    keyword=KeywordExtractor,
)


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

    def add_documents(self, docs: Sequence[Union[Document, TextNode]]):
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

    @staticmethod
    def read_documents(
        path: str,
        show_progres: bool = False,
        num_workers: int = None,
        recursive: bool = False,
        **kwargs,
    ) -> List[Union[Document, TextNode]]:
        """Read documents from a directory.

        Parameters
        ----------
        path: str
            path to the directory containing the documents.
        show_progres: bool, optional, default is False.
            True to show progress bar.
        num_workers: int, optional, default is None.
            The number of workers to use for loading the data.
        recursive: bool, optional, default is False.
            True to read from subdirectories.

        Returns
        -------
        Sequence[Union[Document, TextNode]]
            The documents/nodes read from the store.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")

        reader = SimpleDirectoryReader(path, recursive=recursive, **kwargs)
        documents = reader.load_data(
            show_progress=show_progres, num_workers=num_workers, **kwargs
        )

        for doc in documents:
            # exclude the file name from the llm metadata in order to avoid affecting the llm by weird file names
            doc.excluded_llm_metadata_keys = ["file_name"]
            # exclude the file name from the embeddings metadata in order to avoid affecting the llm by weird file names
            doc.excluded_embed_metadata_keys = ["file_name"]
            # Generate a hash based on the document's text content
            content_hash = generate_content_hash(doc.text)
            doc.doc_id = content_hash  # Assign the hash as the doc_id

        return documents

    @staticmethod
    def extract_info(documents: List[Document], info: Dict[str, Dict[str, int]] = None):
        """Extract Info

        Parameters
        ----------
        documents: List[Document]
            List of documents.
        info: Union[List[str], str], optional, default is None
            The information to extract from the documents.

            >>> info = {
            >>>     "text_splitter": {"separator" : " ", "chunk_size":512, "chunk_overlap":128},
            >>>     "title": {"nodes": 5} ,
            >>>     "question_answer": {"questions": 3},
            >>>     "summary": {"summaries": ["prev", "self"]},
            >>>     "keyword": {"keywords": 10},
            >>>     "entity": {"prediction_threshold": 0.5}
            >>> }

        Returns
        -------
        List[Union[Document, TextNode]]
            The extracted nodes.
        """
        extractors = [
            EXTRACTORS[key](**val) for key, val in info.items() if key in EXTRACTORS
        ]
        pipeline = IngestionPipeline(transformations=extractors)

        nodes = pipeline.run(
            documents=documents,
            in_place=True,
            show_progress=True,
            # num_workers=4
        )
        return nodes
