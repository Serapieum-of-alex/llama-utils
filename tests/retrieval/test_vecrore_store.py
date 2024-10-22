import os
import pytest
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

        test_constructor.add_docs([document, text_node])
        print(test_constructor.store.docstore.get_document("d1"))
        assert len(test_constructor.store.docstore.docs) == 2
        docstore = test_constructor.store.docstore
        assert docstore.get_document("d1") == document
        assert docstore.get_document("d2") == text_node
