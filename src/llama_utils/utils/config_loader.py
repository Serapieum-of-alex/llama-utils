from typing import Any
from llama_index.core import Settings
from llama_utils.utils.models import get_ollama_llm, get_hugging_face_embedding
from llama_index.core.node_parser import SentenceSplitter


TEXT_SPLITTER = SentenceSplitter(chunk_size=1024, chunk_overlap=20)


class ConfigLoader:
    """A class or function to load configuration files (e.g., YAML, JSON)."""

    def __init__(
        self,
        llm: Any = None,
        embedding: Any = None,
    ):
        """Initialize the ConfigLoader class.

        Parameters
        ----------
        llm: Any, optional, default is llama3
            llm model to use.
        embedding: Any, optional, default is BAAI/bge-base-en-v1.5
            Embedding model to use.
        """
        if llm is None:
            llm = get_ollama_llm()
        if embedding is None:
            embedding = get_hugging_face_embedding()

        Settings.embed_model = embedding
        Settings.llm = llm
        Settings.text_splitter = TEXT_SPLITTER
        self._settings = Settings
        self._embedding = embedding
        self._llm = llm

    @property
    def settings(self):
        return self._settings

    @property
    def llm(self):
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value
        Settings.llm = value

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value
        Settings.embed_model = value
