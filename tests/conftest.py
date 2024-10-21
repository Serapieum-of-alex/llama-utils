import pytest
from llama_index.core.schema import Document, TextNode


@pytest.fixture()
def document() -> Document:
    return Document(text="my test document", id_="d1", metadata={"foo": "bar"})


@pytest.fixture()
def text_node() -> TextNode:
    return TextNode(text="my test text node", id_="d2", metadata={"node": "info"})
