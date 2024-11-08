"""A module for managing vector Storage and retrieval."""

import os
from pathlib import Path
from typing import Sequence, Union, List, Dict
import pandas as pd
from llama_index.core.storage.docstore import SimpleDocumentStore, BaseDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core import StorageContext
from llama_index.core.schema import Document, TextNode, BaseNode
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
    SummaryExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_utils.utils.config_loader import ConfigLoader
from llama_utils.utils.helper_functions import generate_content_hash
from llama_utils.utils.errors import StorageNotFoundError

ConfigLoader()

EXTRACTORS = dict(
    text_splitter=TokenTextSplitter,
    title=TitleExtractor,
    question_answer=QuestionsAnsweredExtractor,
    summary=SummaryExtractor,
    keyword=KeywordExtractor,
)
ID_MAPPING_FILE = "metadata_index.csv"


class Storage:
    """A class to manage vector Storage and retrieval."""

    def __init__(
        self,
        storage_backend: Union[str, StorageContext] = None,
        metadata_index: pd.DataFrame = None,
    ):
        """Initialize the Storage.

        Parameters
        ----------
        storage_backend: str, optional, default=None
            The desired vector Storage backend (e.g., Qdrant, FAISS). If none is provided, a simple Storage context
            will be created.
        """
        if not isinstance(storage_backend, StorageContext):
            raise ValueError(
                "Storage class should be instantiated using StorageContext object, given: {storage_backend}"
            )

        self._store = storage_backend
        if isinstance(metadata_index, pd.DataFrame):
            self._metadata_index = metadata_index
        elif metadata_index is None:
            self._metadata_index = create_metadata_index_existing_docs(
                self._store.docstore.docs
            )
        else:
            raise ValueError(
                f"Invalid Storage backend: {storage_backend}. Must be a string or StorageContext."
            )

    @classmethod
    def create(cls) -> "Storage":
        """Create a new instance of the Storage class."""
        storage = cls._create_simple_storage_context()
        metadata_index = cls._create_metadata_index()
        return cls(storage, metadata_index)

    @staticmethod
    def _create_simple_storage_context() -> StorageContext:
        """Create a simple Storage context."""
        return StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=SimpleVectorStore(),
            index_store=SimpleIndexStore(),
        )

    @staticmethod
    def _create_metadata_index():
        """Create a metadata-based index."""
        """Create a metadata-based index."""
        return pd.DataFrame(columns=["file_name", "doc_id"])

    @property
    def store(self) -> StorageContext:
        """Get the Storage context."""
        return self._store

    @property
    def docstore(self) -> BaseDocumentStore:
        """Get the document store."""
        return self.store.docstore

    @property
    def vector_store(self):
        return self.store.vector_store

    @property
    def index_store(self) -> BaseIndexStore:
        return self.store.index_store

    def save(self, store_dir: str):
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
        file_path = os.path.join(store_dir, ID_MAPPING_FILE)
        save_metadata_index(self.metadata_index, file_path)

    @classmethod
    def load(cls, store_dir: str) -> "Storage":
        """Load the store from a directory.

        Parameters
        ----------
        store_dir: str
            The directory containing the store.

        Returns
        -------
        None
        """
        if not Path(store_dir).exists():
            StorageNotFoundError(f"Storage not found at {store_dir}")
        storage = StorageContext.from_defaults(persist_dir=store_dir)
        metadata_index = read_metadata_index(path=store_dir)
        return cls(storage, metadata_index)

    @property
    def metadata_index(self) -> pd.DataFrame:
        """Get the metadata index."""
        return self._metadata_index

    def add_documents(
        self,
        docs: Sequence[Union[Document, TextNode]],
        generate_id: bool = True,
        update: bool = False,
    ):
        """Add node/documents to the store.

            The `add_documents` method adds a node to the store. The node's id is a sha256 hash generated based on the
            node's text content. if the `update` parameter is True and the nodes already exist the existing node will
            be updated.

        Parameters
        ----------
        docs: Sequence[TextNode/Document]
            The node/documents to add to the store.
        generate_id: bool, optional, default is False.
            True if you want to generate a sha256 hash number as a doc_id based on the content of the nodes
        update: bool, optional, default is True.
            True to update the document in the docstore if it already exist.

        Returns
        -------
        None
        """
        new_entries = []
        file_names = []
        # Create a metadata-based index
        for doc in docs:
            # change the id to a sha256 hash if it is not already
            if generate_id:
                doc.node_id = generate_content_hash(doc.text)

            if not self.docstore.document_exists(doc.node_id) or update:
                self.docstore.add_documents([doc], allow_update=update)
                # Update the metadata index with file name as key and doc_id as value
                file_name = os.path.basename(doc.metadata["file_path"])
                if file_name in file_names:
                    file_name = f"{file_name}_{len(file_names)}"
                new_entries.append({"file_name": file_name, "doc_id": doc.node_id})
                file_names.append(file_name)
            else:
                print(f"Document with ID {doc.node_id} already exists. Skipping.")

        # Convert new entries to a DataFrame and append to the existing metadata DataFrame
        if new_entries:
            new_entries_df = pd.DataFrame(new_entries)
            self._metadata_index = pd.concat(
                [self._metadata_index, new_entries_df], ignore_index=True
            )

    @staticmethod
    def read_documents(
        path: str,
        show_progres: bool = False,
        num_workers: int = None,
        recursive: bool = False,
        **kwargs,
    ) -> List[Union[Document, TextNode]]:
        """Read documents from a directory.

        the `read_documents` method reads documents from a directory and returns a list of documents.
        the `doc_id` is sha256 hash number generated based on the document's text content.

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
            # Assign the hash as the doc_id
            doc.doc_id = content_hash

        return documents

    def get_nodes_by_file_name(
        self, file_name: str, exact_match: bool = False
    ) -> List[BaseNode]:
        """Get nodes by file name.

        Parameters
        ----------
        file_name: str
            The file name to search for.
        exact_match: bool, optional, default is False
            True to search for an exact match, False to search for a partial match.

        Returns
        -------
        List[TextNode]
            The nodes with the specified file name.
        """
        if exact_match:
            doc_ids = self.metadata_index.loc[
                self.metadata_index["file_name"] == file_name, "doc_id"
            ].values
        else:
            doc_ids = self.metadata_index.loc[
                self.metadata_index["file_name"].str.contains(file_name, regex=True),
                "doc_id",
            ].values
        docs = self.docstore.get_nodes(doc_ids)
        return docs

    @staticmethod
    def extract_info(
        documents: List[Union[Document, BaseNode]],
        info: Dict[str, Dict[str, int]] = None,
    ) -> List[TextNode]:
        """Extract Info

        Parameters
        ----------
        documents: List[Union[Document, BaseNode]]
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
        List[TextNode]
            The extracted nodes.
            title:
                the extracted title will be stored in the metadata under the key "document_title".
            question_answer:
                the extracted questions will be stored in the metadata under the key "questions_this_excerpt_can_answer".
            summary:
                the extracted summaries will be stored in the metadata under the key "summary".
            keyword:
                the extracted keywords will be stored in the metadata under the key "keywords".
            entity:
                the extracted entities will be stored in the metadata under the key "entities".
        """
        info = EXTRACTORS.copy() if info is None else info

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


def read_metadata_index(path: str) -> pd.DataFrame:
    """Read the ID mapping from a JSON file."""
    file_path = os.path.join(path, ID_MAPPING_FILE)
    data = pd.read_csv(file_path, index_col=0)
    return data


def save_metadata_index(data: pd.DataFrame, path: str):
    """Save the ID mapping to a JSON file."""
    data.to_csv(path, index=True)


def create_metadata_index_existing_docs(docs: Dict[str, BaseNode]):
    metadata_index = {}
    i = 0
    for key, val in docs.items():
        metadata_index[i] = {
            "file_name": val.metadata["file_name"],
            "doc_id": generate_content_hash(val.text),
        }
        i += 1
    df = pd.DataFrame.from_dict(metadata_index, orient="index")
    return df
