import os
from typing import Any
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_utils import __path__

TEXT_SPLITTER = SentenceSplitter(chunk_size=1024, chunk_overlap=20)


class ConfigLoader:
    """A class or function to load configuration files (e.g., YAML, JSON)."""

    def __init__(
        self,
        model_id: str = "llama3",
        model_dir: str = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm: Any = None,
        embedding: Any = None,
    ):
        """Initialize the ConfigLoader class.

        Parameters
        ----------
        model_id : str, optional
            The model ID, by default "llama3"
        model_dir : str, optional, default is None
            The model directory.
        embedding_model : str, optional
        """
        if model_dir is None:
            model_dir = os.path.join(__path__[0], "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        if llm is None:
            self._llm = Ollama(model=model_id, request_timeout=360.0)
        if embedding is None:
            self._embedding = HuggingFaceEmbedding(
                model_name=embedding_model, cache_folder=model_dir
            )
        Settings.embed_model = self._embedding
        Settings.llm = self._llm
        Settings.text_splitter = TEXT_SPLITTER
        self._settings = Settings

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
