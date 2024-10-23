import os
import pytest
from unittest.mock import patch, MagicMock
from llama_index.core.schema import Document, TextNode
from llama_utils.retrieval.vector_store import VectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext


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
    def test_constructor(self) -> VectorStore:
        vector_store = VectorStore()
        assert vector_store is not None, "VectorStore not created."
        assert (
            isinstance(vector_store.store, StorageContext) is not None
        ), "Storage context not created."
        return vector_store

    def test_save_store(self, test_constructor: VectorStore):
        path = "tests/data/store"
        test_constructor.save_store(path)
        assert os.path.exists(path), "Store not saved."
        docstore_content = [
            "default__vector_store.json",
            "docstore.json",
            "graph_store.json",
            "image__vector_store.json",
            "index_store.json",
        ]
        assert all(elem in os.listdir(path) for elem in docstore_content)

    def test_load_store(self, test_constructor: VectorStore):
        path = "tests/data/load_store"
        test_constructor.load_store(path)
        assert isinstance(test_constructor.store, StorageContext)

    def test_add_docs(
        self, test_constructor: VectorStore, document: Document, text_node: TextNode
    ):

        test_constructor.add_documents([document, text_node])
        print(test_constructor.store.docstore.get_document("d1"))
        assert len(test_constructor.store.docstore.docs) == 2
        docstore = test_constructor.store.docstore
        assert docstore.get_document("d1") == document
        assert docstore.get_document("d2") == text_node


def test_read_documents(data_path: str):
    docs = VectorStore.read_documents(data_path)
    assert len(docs) == 4
    doc = docs[0]
    assert doc.excluded_embed_metadata_keys == ["file_name"]
    assert doc.excluded_embed_metadata_keys == ["file_name"]


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
