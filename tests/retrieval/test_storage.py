import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.schema import Document, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

from llama_utils.retrieval.storage import Storage
from llama_utils.utils.helper_functions import generate_content_hash


def test_create_simple_storage_context():
    storage_context = Storage._create_simple_storage_context()
    assert isinstance(storage_context, StorageContext), "Storage context not created."
    assert list(storage_context.vector_stores.keys()) == ["default", "image"]
    assert (
        storage_context.vector_stores["default"] is not None
    ), "Default vector Storage not created."
    assert (
        storage_context.vector_stores["image"] is not None
    ), "Image vector Storage not created."
    assert isinstance(storage_context.docstore, SimpleDocumentStore)
    assert isinstance(storage_context.index_store, SimpleIndexStore)
    assert isinstance(storage_context.vector_store, SimpleVectorStore)
    assert isinstance(storage_context.graph_store, SimpleGraphStore)


class TestStorage:

    @pytest.fixture
    def test_empty_storage(self) -> Storage:
        store = Storage.create()
        assert store is not None, "Storage not created."
        assert (
            isinstance(store.store, StorageContext) is not None
        ), "Storage context not created."
        assert isinstance(store.metadata_index, pd.DataFrame)
        return store

    def test_properties(self, test_empty_storage: Storage):
        isinstance(test_empty_storage.docstore, SimpleDocumentStore)
        isinstance(test_empty_storage.vector_store, SimpleVectorStore)
        isinstance(test_empty_storage.index_store, SimpleIndexStore)

    def test_load(self, storage_path: str):
        store = Storage.load(storage_path)
        storage = store._store
        assert isinstance(storage, StorageContext)
        assert isinstance(storage.docstore, SimpleDocumentStore)
        assert isinstance(store.metadata_index, pd.DataFrame)
        assert len(storage.docstore.docs) == 4

    def test_storage_context(self, storage_docstore: StorageContext):
        store = Storage(storage_docstore)
        storage = store._store
        assert isinstance(storage, StorageContext)
        assert isinstance(storage.docstore, SimpleDocumentStore)
        assert isinstance(store.metadata_index, pd.DataFrame)
        assert len(storage.docstore.docs) == 4

    def test_constructor_raise_error(self):
        with pytest.raises(ValueError):
            Storage(5)

    def test_save(self, test_empty_storage: Storage):
        path = "tests/data/Storage"
        test_empty_storage.save(path)
        assert os.path.exists(path), "Storage not saved."
        docstore_content = [
            "default__vector_store.json",
            "docstore.json",
            "graph_store.json",
            "image__vector_store.json",
            "index_store.json",
        ]
        assert all(elem in os.listdir(path) for elem in docstore_content)

    def test_add_documents(
        self,
        test_empty_storage: Storage,
        document: Document,
        text_node: TextNode,
        hash_document: str,
        hash_text_node: str,
    ):
        test_empty_storage.add_documents([document, text_node])
        assert len(test_empty_storage.store.docstore.docs) == 2
        docstore = test_empty_storage.store.docstore
        assert docstore.get_document(hash_document) == document
        assert docstore.get_document(hash_text_node) == text_node
        df = test_empty_storage.metadata_index
        assert df.shape[0] == 2
        assert df.loc[0, "doc_id"] == hash_document
        assert df.loc[1, "doc_id"] == hash_text_node

    def test_add_duplicated_documents(
        self,
        capsys,
        test_empty_storage: Storage,
        document: Document,
        text_node: TextNode,
        hash_document: str,
        hash_text_node: str,
    ):
        test_empty_storage.add_documents([document, text_node])
        test_empty_storage.add_documents([document, text_node])
        # capture the printed text
        captured = capsys.readouterr()
        assert len(test_empty_storage.store.docstore.docs) == 2
        assert captured.out == (
            "Document with ID 8323ac870e04bcf4b64eb04624001a025027d8f797414072df1b81e087f74fb3 "
            "already exists. Skipping.\nDocument with ID "
            "dfbab7917ff16a68316aaf745bbbaeffe4b8c1692763548605020c227831c1c4 already exists. Skipping.\n"
        )

    def test_different_nodes_same_document(
        self,
        test_empty_storage: Storage,
        text_node_2: TextNode,
        text_node: TextNode,
        hash_text_node: str,
    ):
        """
        Different nodes with the same document id should be added to the Storage.

        The test check if the file_name is added in the metadata index with an incremented index.
        <FILE-NAME>-1, <FILE-NAME>-2, ...
        """
        test_empty_storage.add_documents([text_node, text_node_2])
        assert len(test_empty_storage.store.docstore.docs) == 2
        docstore = test_empty_storage.store.docstore
        assert docstore.get_document(hash_text_node) == text_node
        df = test_empty_storage.metadata_index
        assert df.loc[:, "file_name"].to_list() == ["node-path", "node-path_1"]

    def test_get_nodes_by_file_name(
        self,
        test_empty_storage: Storage,
        text_node_2: TextNode,
        text_node: TextNode,
    ):
        test_empty_storage.add_documents([text_node, text_node_2])
        nodes = test_empty_storage.get_nodes_by_file_name("node-")
        assert nodes == [text_node, text_node_2]
        nodes = test_empty_storage.get_nodes_by_file_name("node-path", exact_match=True)
        assert nodes == [text_node]

    def test_node_id_list(
        self,
        test_empty_storage: Storage,
        text_node_2: TextNode,
        text_node: TextNode,
    ):
        """docstore has only two text nodes and no documents."""
        test_empty_storage.add_documents([text_node, text_node_2])
        node_list = test_empty_storage.node_id_list()
        assert node_list == [
            "dfbab7917ff16a68316aaf745bbbaeffe4b8c1692763548605020c227831c1c4",
            "cc385eb9c8562d248624152b09f90c366b441d9e1f8d0f3752aca2124cb36dd7",
        ]

    def test_document_metadata(self, paul_graham_essay_storage: Storage):

        document_metadata = paul_graham_essay_storage.document_metadata
        assert isinstance(document_metadata, dict)
        assert len(document_metadata.keys()) == 1
        doc_metadata = document_metadata[list(document_metadata.keys())[0]]
        assert list(doc_metadata.metadata.keys()) == [
            "file_path",
            "file_name",
            "file_type",
            "file_size",
            "creation_date",
            "last_modified_date",
            "document_title",
        ]
        assert len(doc_metadata.node_ids) == 53

    def test_delete_document(
        self, paul_graham_essay_storage: Storage, essay_document_id: str
    ):
        paul_graham_essay_storage.delete_document(essay_document_id)
        assert (
            essay_document_id not in paul_graham_essay_storage.document_metadata.keys()
        )


def test_read_documents(data_path: str):
    docs = Storage.read_documents(data_path)
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
    nodes = Storage.apply_extractors(documents, info)

    # Check if the pipeline.run method was called with expected arguments
    mock_pipeline_run.assert_called_once_with(
        documents=documents,
        in_place=True,
        show_progress=True,
        # num_workers=4 (optional if you have it)
    )
    assert mock_pipeline_run.call_count == 1
    assert nodes == mock_pipeline_run.return_value
