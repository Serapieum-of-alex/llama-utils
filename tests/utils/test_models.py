import pytest
from llama_index.llms.ollama import Ollama
from llama_utils.utils.models import get_ollama_llm, azure_open_ai


def test_get_ollama_llm():
    llm = get_ollama_llm()
    assert isinstance(llm, Ollama)
    assert llm.context_window == 3900
    assert llm.model == "llama3"
    assert llm.request_timeout == 360.0
    assert llm.temperature == 0.75
    assert llm is not None


def test_azure_open_ai():
    with pytest.warns(
        UserWarning, match="Azure OpenAI environment variables are not set."
    ):
        llm = azure_open_ai()
        assert llm.temperature == 0
        assert llm.model == "gpt-4o"
        assert llm.engine == "4o"
