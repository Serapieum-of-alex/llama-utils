from llama_index.llms.ollama import Ollama
from llama_index.core.settings import _Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_utils.utils.config_loader import ConfigLoader, TEXT_SPLITTER


def test_config_loader():
    config = ConfigLoader()
    assert isinstance(config.llm, Ollama)
    assert isinstance(config.settings, _Settings)
    assert isinstance(config.embedding, HuggingFaceEmbedding)
    assert config.settings.llm == config.llm
    assert config.settings.embed_model == config.embedding
    assert config.settings.text_splitter == TEXT_SPLITTER
