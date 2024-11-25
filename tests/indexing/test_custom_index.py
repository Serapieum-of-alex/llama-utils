import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores import SimpleVectorStore

from llama_utils.indexing.custom_index import CustomIndex


def test_create_from_document(document: Document):
    content_id = [
        "8323ac870e04bcf4b64eb04624001a025027d8f797414072df1b81e087f74fb3",
        "06c83efcd22e3b755ca95ffe22954a304740027769fb541a126a5d23b6f5ac1f",
    ]
    doc_list = [document, "any text"]
    index = CustomIndex.create_from_documents(doc_list)
    assert isinstance(index, CustomIndex)
    assert (
        index.__str__()
        == f"\n        Index ID: {index.id}\n        Number of Document: {len(index.doc_ids)}\n        "
    )
    assert isinstance(index.index, VectorStoreIndex)
    assert isinstance(index.id, str)
    assert isinstance(index.vector_store, SimpleVectorStore)
    assert index.doc_ids == content_id
    assert isinstance(index.embeddings, dict)
    assert list(index.embeddings.keys()) == content_id
    # the embedding is a list of 768 floats based on the BERT model
    assert len(index.embeddings[content_id[0]]) == 768


def test_create_from_nodes(text_node: TextNode):
    index = CustomIndex.create_from_nodes([text_node])
    assert isinstance(index, CustomIndex)
    assert len(index.embeddings) == 1
    assert list(index.index.docstore.docs.keys()) == [text_node.id_]


def test_create_custome_index_wrong_input():
    with pytest.raises(ValueError):
        CustomIndex("wrong input")
