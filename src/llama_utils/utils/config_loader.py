from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

TEXT_SPLITTER = SentenceSplitter(chunk_size=1024, chunk_overlap=20)


class ConfigLoader:
    """A class or function to load configuration files (e.g., YAML, JSON)."""

    def __init__(
        self,
        model_id: str = "llama3",
        model_dir: str = r"C:\MyComputer\llm\models",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self._llm = Ollama(model=model_id, request_timeout=360.0)
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

    @property
    def embedding(self):
        return self._embedding
