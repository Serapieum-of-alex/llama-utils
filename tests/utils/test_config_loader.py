from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import _Settings

from llama_utils.utils.config_loader import ConfigLoader


def test_config_loader():
    mock_embedding = MockEmbedding(embed_dim=768)
    llm = MockLLM()
    config = ConfigLoader(llm=llm, embedding=mock_embedding)
    assert isinstance(config.llm, MockLLM)
    assert isinstance(config.settings, _Settings)
    assert isinstance(config.embedding, MockEmbedding)
    assert config.settings.llm == config.llm
    assert config.settings.embed_model == config.embedding
    assert isinstance(config.settings.text_splitter, SentenceSplitter)
