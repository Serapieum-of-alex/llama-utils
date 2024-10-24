import os
import pytest
import pandas as pd
from llama_utils.utils.helper_functions import generate_content_hash
from unittest.mock import patch, MagicMock
from llama_index.core.schema import Document, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext

from llama_utils.retrieval.vector_store import VectorStore


def test_create_storage_context():
    # vector_store = VectorStore()
    storage_context = VectorStore._create_simple_storage_context()
    assert isinstance(storage_context, StorageContext), "Storage context not created."
    assert list(storage_context.vector_stores.keys()) == ["default", "image"]
    assert (
        storage_context.vector_stores["default"] is not None
    ), "Default vector store not created."
    assert (
        storage_context.vector_stores["image"] is not None
    ), "Image vector store not created."
    assert isinstance(storage_context.docstore, SimpleDocumentStore)
    assert isinstance(storage_context.index_store, SimpleIndexStore)
    assert isinstance(storage_context.vector_store, SimpleVectorStore)
    assert isinstance(storage_context.graph_store, SimpleGraphStore)


class TestVectorStore:

    @pytest.fixture
    def test_constructor_no_storage(self) -> VectorStore:
        vector_store = VectorStore()
        assert vector_store is not None, "VectorStore not created."
        assert (
            isinstance(vector_store.store, StorageContext) is not None
        ), "Storage context not created."
        assert isinstance(vector_store.metadata_index, pd.DataFrame)
        return vector_store

    def test_constructor_storage_path(self, storage_path: str):
        vector_store = VectorStore(storage_path)
        store = vector_store._store
        assert isinstance(store, StorageContext)
        assert isinstance(store.docstore, SimpleDocumentStore)
        assert isinstance(vector_store.metadata_index, pd.DataFrame)
        assert len(store.docstore.docs) == 4

    def test_constructor_storage_context(self, storage_docstore: StorageContext):
        vector_store = VectorStore(storage_docstore)
        store = vector_store._store
        assert isinstance(store, StorageContext)
        assert isinstance(store.docstore, SimpleDocumentStore)
        assert isinstance(vector_store.metadata_index, pd.DataFrame)
        assert len(store.docstore.docs) == 4

    def test_constructor_raise_error(self):
        with pytest.raises(ValueError):
            VectorStore(5)

    def test_save_store(self, test_constructor_no_storage: VectorStore):
        path = "tests/data/store"
        test_constructor_no_storage.save_store(path)
        assert os.path.exists(path), "Store not saved."
        docstore_content = [
            "default__vector_store.json",
            "docstore.json",
            "graph_store.json",
            "image__vector_store.json",
            "index_store.json",
        ]
        assert all(elem in os.listdir(path) for elem in docstore_content)

    def test_load_store(self, test_constructor_no_storage: VectorStore):
        # empty store
        path = "tests/data/load_store"
        test_constructor_no_storage.load_store(path)
        assert isinstance(test_constructor_no_storage.store, StorageContext)

    def test_add_documents(
        self,
        test_constructor_no_storage: VectorStore,
        document: Document,
        text_node: TextNode,
        hash_document: str,
        hash_text_node: str,
    ):
        test_constructor_no_storage.add_documents([document, text_node])
        assert len(test_constructor_no_storage.store.docstore.docs) == 2
        docstore = test_constructor_no_storage.store.docstore
        assert docstore.get_document(hash_document) == document
        assert docstore.get_document(hash_text_node) == text_node
        df = test_constructor_no_storage.metadata_index
        assert df.shape[0] == 2
        assert df.loc[0, "doc_id"] == hash_document
        assert df.loc[1, "doc_id"] == hash_text_node

    def test_different_nodes_same_document(
        self,
        test_constructor_no_storage: VectorStore,
        text_node_2: TextNode,
        text_node: TextNode,
        hash_text_node: str,
    ):
        """
        Different nodes with the same document id should be added to the store.

        The test check if the file_name is added in the metadata index with an incremented index.
        <FILE-NAME>-1, <FILE-NAME>-2, ...
        """
        test_constructor_no_storage.add_documents([text_node, text_node_2])
        assert len(test_constructor_no_storage.store.docstore.docs) == 2
        docstore = test_constructor_no_storage.store.docstore
        assert docstore.get_document(hash_text_node) == text_node
        df = test_constructor_no_storage.metadata_index
        assert df.loc[:, "file_name"].to_list() == ["node-path", "node-path_1"]

    def test_get_nodes_by_file_name(
        self,
        test_constructor_no_storage: VectorStore,
        text_node_2: TextNode,
        text_node: TextNode,
    ):
        test_constructor_no_storage.add_documents([text_node, text_node_2])
        nodes = test_constructor_no_storage.get_nodes_by_file_name("node-")
        assert nodes == [text_node, text_node_2]
        nodes = test_constructor_no_storage.get_nodes_by_file_name(
            "node-path", exact_match=True
        )
        assert nodes == [text_node]


def test_read_documents(data_path: str):
    docs = VectorStore.read_documents(data_path)
    assert len(docs) == 4
    doc = docs[0]
    assert doc.excluded_embed_metadata_keys == ["file_name"]
    assert doc.excluded_embed_metadata_keys == ["file_name"]
    assert docs[0].doc_id == generate_content_hash(docs[0].text)


@patch("llama_index.core.ingestion.IngestionPipeline.run")
def test_extract_info(mock_pipeline_run, document: Document, text_node: TextNode):
    documents = [document, text_node]

    # Set up the mock for the pipeline instance
    mock_pipeline_run.return_value = MagicMock(return_value="mocked_nodes")

    info = {
        "text_splitter": {"separator": " ", "chunk_size": 512, "chunk_overlap": 128},
        "title": {"nodes": 5},
        "question_answer": {"questions": 3},
        "summary": {"summaries": ["prev", "self"]},
        "keyword": {"keywords": 10},
        "entity": {"prediction_threshold": 0.5},
    }
    nodes = VectorStore.extract_info(documents, info)

    # Check if the pipeline.run method was called with expected arguments
    mock_pipeline_run.assert_called_once_with(
        documents=documents,
        in_place=True,
        show_progress=True,
        # num_workers=4 (optional if you have it)
    )
    assert mock_pipeline_run.call_count == 1
    assert nodes == mock_pipeline_run.return_value
