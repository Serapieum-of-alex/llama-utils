import unittest
from unittest.mock import MagicMock, patch

import pytest
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_utils.utils.models import (
    DEFAULT_CONTEXT_WINDOW,
    LLMModel,
    azure_open_ai,
    get_hugging_face_embedding,
    get_ollama_llm,
)


@pytest.mark.mock
def test_get_ollama_llm():
    with patch("llama_index.llms.ollama.Ollama") as mock_ollama:
        instance = MagicMock()
        mock_ollama.return_value = instance

        llm = get_ollama_llm()
        assert llm == instance
        mock_ollama.assert_called_once_with(
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.75,
            context_window=DEFAULT_CONTEXT_WINDOW,
            request_timeout=360.0,
            prompt_key="prompt",
            json_mode=False,
            additional_kwargs={},
            is_function_calling_model=True,
            keep_alive=None,
        )


@pytest.mark.optional_package
def test_azure_open_ai():
    with pytest.warns(
        UserWarning, match="Azure OpenAI environment variables are not set."
    ):
        llm = azure_open_ai()
        assert llm.temperature == 0
        assert llm.model == "gpt-4o"
        assert llm.engine == "4o"


@pytest.mark.optional_package
def test_embedding():
    model = get_hugging_face_embedding(model_name="BAAI/bge-small-en-v1.5")
    assert isinstance(model, HuggingFaceEmbedding)
    assert model.model_name == "BAAI/bge-small-en-v1.5"
    assert model.cache_folder is None
    assert model.max_length == 512


class TestGetOllamaLLM(unittest.TestCase):

    @pytest.mark.mock
    @patch("llama_index.llms.ollama.Ollama")
    def test_get_ollama_llm_defaults(self, mock_ollama):
        """Test get_ollama_llm with default parameters."""
        instance = MagicMock()
        mock_ollama.return_value = instance

        result = get_ollama_llm(model_id="llama3")

        mock_ollama.assert_called_once_with(
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.75,
            context_window=DEFAULT_CONTEXT_WINDOW,
            request_timeout=360.0,
            prompt_key="prompt",
            json_mode=False,
            additional_kwargs={},
            is_function_calling_model=True,
            keep_alive=None,
        )
        self.assertEqual(result, instance)


class TestLLMModel(unittest.TestCase):

    @pytest.mark.mock
    @patch("llama_utils.utils.models.azure_open_ai")
    def test_initialize_azure_model(self, mock_azure_open_ai):
        """Test LLMModel initialization with Azure OpenAI."""
        mock_instance = MagicMock()
        mock_azure_open_ai.return_value = mock_instance

        model = LLMModel(model_type="azure", model_id="gpt-4o", engine="4o")

        mock_azure_open_ai.assert_called_once_with(model_id="gpt-4o", engine="4o")
        self.assertEqual(model.base_model, mock_instance)

    @pytest.mark.mock
    @patch("llama_utils.utils.models.get_ollama_llm")
    def test_initialize_ollama_model(self, mock_get_ollama_llm):
        """Test LLMModel initialization with Ollama."""
        mock_instance = MagicMock()
        mock_get_ollama_llm.return_value = mock_instance

        model = LLMModel(
            model_type="ollama", model_id="llama3", temperature=0.5, context_window=1024
        )

        mock_get_ollama_llm.assert_called_once_with(
            model_id="llama3", temperature=0.5, context_window=1024
        )
        self.assertEqual(model.base_model, mock_instance)

    @pytest.mark.mock
    @patch("llama_index.llms.huggingface.HuggingFaceLLM")
    def test_initialize_huggingface_model(self, mock_huggingface_llm):
        """Test LLMModel initialization with HuggingFace."""
        mock_instance = MagicMock()
        mock_huggingface_llm.return_value = mock_instance

        model = LLMModel(model_type="huggingface", model_name="distilgpt2")

        mock_huggingface_llm.assert_called_once()
        self.assertEqual(model.base_model, mock_instance)

    @pytest.mark.mock
    @patch("llama_utils.utils.models.get_ollama_llm")
    def test_generate_response_ollama(self, mock_get_ollama_llm):
        """Test LLMModel generate_response with Ollama."""
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "Mocked response"
        mock_get_ollama_llm.return_value = mock_instance

        model = LLMModel(model_type="ollama", model_id="llama3")
        response = model.generate_response("Test prompt")

        mock_instance.complete.assert_called_once_with("Test prompt")
        self.assertEqual(response, "Mocked response")

    @pytest.mark.mock
    @patch("llama_utils.utils.models.azure_open_ai")
    def test_generate_response_azure(self, mock_azure_open_ai):
        """Test LLMModel generate_response with Azure OpenAI."""
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "Azure response"
        mock_azure_open_ai.return_value = mock_instance

        model = LLMModel(model_type="azure", model_id="gpt-4o")
        response = model.generate_response("Test prompt")

        mock_instance.complete.assert_called_once()
        self.assertEqual(response, "Azure response")
