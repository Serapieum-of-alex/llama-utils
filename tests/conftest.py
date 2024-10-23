import pytest
from llama_index.core.schema import Document, TextNode
from llama_index.core import StorageContext


@pytest.fixture()
def document() -> Document:
    return Document(text="my test document", id_="d1", metadata={"foo": "bar"})


@pytest.fixture()
def text_node() -> TextNode:
    return TextNode(text="my test text node", id_="d2", metadata={"node": "info"})


@pytest.fixture()
def data_path() -> str:
    return "tests/data/files"


@pytest.fixture()
def storage_path() -> str:
    return "tests/data/docstore"


@pytest.fixture()
def storage_docstore(storage_path: str) -> StorageContext:
    return StorageContext.from_defaults(persist_dir="tests/data/docstore")
