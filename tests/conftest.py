from pathlib import Path
from typing import Dict

import pytest
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import Document, TextNode

Settings.embed_model = MockEmbedding(embed_dim=768)
Settings.llm = MockLLM()


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


@pytest.fixture
def vector_store_index(text_node: TextNode) -> VectorStoreIndex:
    """Vector store index with one text node."""
    return VectorStoreIndex([text_node])


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
def paul_grahm_essay_storage():
    return "tests/data/paul-graham-essay-storage"


@pytest.fixture()
def essay_document_id() -> str:
    return "a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ffd112463fb98c227092"


@pytest.fixture()
def essay_node_id() -> str:
    return "cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7d83a76956bf22765b7"


@pytest.fixture
def geoscience_pdf() -> Path:
    return Path("tests/data/pdf/geoscience-paper.pdf")


@pytest.fixture
def geoscience_paper_artifacts() -> Dict[str, str]:
    r_dir = "tests/data/pdf/geoscience-paper-parsing-artifacts"
    return {
        "md_file": f"{r_dir}/geoscience-paper.md",
        "images_dir": f"{r_dir}/geoscience-paper_artifacts",
    }
