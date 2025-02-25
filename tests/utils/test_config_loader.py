from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import _Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_utils.utils.config_loader import ConfigLoader


def test_config_loader():
    config = ConfigLoader()
    assert isinstance(config.llm, Ollama)
    assert isinstance(config.settings, _Settings)
    assert isinstance(config.embedding, HuggingFaceEmbedding)
    assert config.settings.llm == config.llm
    assert config.settings.embed_model == config.embedding
    assert isinstance(config.settings.text_splitter, SentenceSplitter)
