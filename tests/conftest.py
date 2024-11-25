import pytest
from llama_index.core import StorageContext
from llama_index.core.schema import Document, TextNode

from llama_utils.retrieval.storage import Storage
from llama_utils.utils.config_loader import ConfigLoader

ConfigLoader()


@pytest.fixture()
def document() -> Document:
    return Document(
        text="my test document",
        id_="d1",
        metadata={"foo": "bar", "file_path": "document-path"},
    )


@pytest.fixture()
def text_node() -> TextNode:
    return TextNode(
        text="my test text node",
        id_="d2",
        metadata={"node": "info", "file_path": "node-path"},
    )


@pytest.fixture()
def text_node_2() -> TextNode:
    return TextNode(
        text="my test text node 2",
        id_="d2",
        metadata={"node": "info", "file_path": "node-path"},
    )


@pytest.fixture()
def hash_document() -> str:
    return "8323ac870e04bcf4b64eb04624001a025027d8f797414072df1b81e087f74fb3"


@pytest.fixture()
def hash_text_node() -> str:
    return "dfbab7917ff16a68316aaf745bbbaeffe4b8c1692763548605020c227831c1c4"


@pytest.fixture()
def data_path() -> str:
    return "tests/data/files"


@pytest.fixture()
def storage_path() -> str:
    return "tests/data/docstore"


@pytest.fixture()
def storage_docstore(storage_path: str) -> StorageContext:
    return StorageContext.from_defaults(persist_dir="tests/data/docstore")


@pytest.fixture()
def paul_grahm_essay_storage():
    return "tests/data/paul-graham-essay-storage"


@pytest.fixture(scope="function")
def paul_graham_essay_storage(paul_grahm_essay_storage: str) -> Storage:
    return Storage.load(paul_grahm_essay_storage)


@pytest.fixture()
def essay_document_id() -> str:
    return "a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ffd112463fb98c227092"
