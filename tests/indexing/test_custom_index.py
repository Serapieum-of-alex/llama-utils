import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document

from llama_utils.indexing.custom_index import CustomIndex


def test_create_from_document(document: Document):
    doc_list = [document, "any text"]
    index = CustomIndex.create_from_documents(doc_list)
    assert isinstance(index, CustomIndex)
    assert isinstance(index.index, VectorStoreIndex)


def test_create_custome_index_wrong_input():
    with pytest.raises(ValueError):
        CustomIndex("wrong input")
