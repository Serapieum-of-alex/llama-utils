import os
from pathlib import Path
import pytest
from llama_utils.utils.helper_functions import generate_content_hash
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
    def test_constructor_no_storage(self) -> VectorStore:
        vector_store = VectorStore()
        assert vector_store is not None, "VectorStore not created."
        assert (
            isinstance(vector_store.store, StorageContext) is not None
        ), "Storage context not created."
        return vector_store

    def test_constructor_storage_path(self, storage_path: str):
        vector_store = VectorStore(storage_path)
        store = vector_store._store
        assert isinstance(store, StorageContext)
        assert isinstance(store.docstore, SimpleDocumentStore)
        assert len(store.docstore.docs) == 4

    def test_constructor_storage_context(
        self, storage_docstore: StorageContext
    ) -> VectorStore:
        vector_store = VectorStore(storage_docstore)
        store = vector_store._store
        assert isinstance(store, StorageContext)
        assert isinstance(store.docstore, SimpleDocumentStore)
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

    def test_add_docs(
        self,
        test_constructor_no_storage: VectorStore,
        document: Document,
        text_node: TextNode,
    ):

        test_constructor_no_storage.add_documents([document, text_node])
        assert len(test_constructor_no_storage.store.docstore.docs) == 2
        docstore = test_constructor_no_storage.store.docstore
        assert (
            docstore.get_document(
                "8323ac870e04bcf4b64eb04624001a025027d8f797414072df1b81e087f74fb3"
            )
            == document
        )
        assert (
            docstore.get_document(
                "dfbab7917ff16a68316aaf745bbbaeffe4b8c1692763548605020c227831c1c4"
            )
            == text_node
        )


@pytest.fixture
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
