"""LLMs and embedding models."""

import os
from typing import Any, Dict, Optional, Union
from warnings import warn

DEFAULT_HUGGINGFACE_MODEL = "StabilityAI/stablelm-tuned-alpha-3b"
DEFAULT_CONTEXT_WINDOW = 3900
DEFAULT_NUM_OUTPUTS = 256


def azure_open_ai(model_id: str = "gpt-4o", engine: str = "4o"):
    """Get the Azure OpenAI model.

    Parameters
    ----------
    model_id: str, optional, default is "gpt-4o"
        The model ID.
    engine: str, optional, default is "4o"
        The engine.

    Returns
    -------
    AzureOpenAI
        The Azure OpenAI model.

    Raises
    ------
    ImportError
        If the `llama-index-llms-azure-openai` package is not installed.

    Examples
    --------
    >>> from llama_utils.utils.models import azure_open_ai
    >>> from dotenv import load_dotenv
    >>> load_dotenv() # doctest: +SKIP
    >>> llm = azure_open_ai() # doctest: +SKIP
    >>> print(llm.model) # doctest: +SKIP
    gpt-4o
    >>> print(llm.metadata) # doctest: +SKIP
    context_window=128000 num_output=-1 is_chat_model=True is_function_calling_model=True model_name='gpt-4o' system_role=<MessageRole.SYSTEM: 'system'>
    """
    try:
        from llama_index.llms.azure_openai import AzureOpenAI
    except ImportError:
        raise ImportError(
            "Please install the `llama-index-llms-azure-openai` package to use the Azure OpenAI model."
        )
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    if endpoint is None or api_key is None or api_version is None:
        warn("Azure OpenAI environment variables are not set.")

    llm = AzureOpenAI(
        engine="4o" if engine is None else engine,
        model="gpt-4o" if model_id is None else model_id,  # o1-preview
        temperature=0.0,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    return llm


def get_ollama_llm(
    model_id: str = "llama3",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.75,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    request_timeout: float = 360.0,
    prompt_key: str = "prompt",
    json_mode: bool = False,
    additional_kwargs: Dict[str, Any] = {},
    is_function_calling_model: bool = True,
    keep_alive: Optional[Union[float, str]] = None,
):
    """Get the Ollama LLM with flexible parameters.

    Parameters
    ----------
    model_id : str
        The model ID to use.
    base_url : str, optional
        The base URL of the Ollama API.
    temperature : float, optional
        The temperature setting for response randomness.
    context_window : int, optional
        Maximum token window for context.
    request_timeout : float, optional
        Timeout for requests.
    prompt_key : str, optional
        Key for the prompt in requests.
    json_mode : bool, optional
        Whether to return responses in JSON mode.
    additional_kwargs : dict, optional
        Additional model-specific parameters.
    is_function_calling_model : bool, optional
        Whether the model supports function calling.
    keep_alive : Optional[Union[float, str]], optional
        Keep-alive duration.

    Returns
    -------
    Ollama
        An instance of the Ollama LLM.

    Raises
    ------
    ImportError
        If the `llama-index-llms-ollama` package is not installed.

    Examples
    --------
    >>> from llama_utils.utils.models import get_ollama_llm
    >>> llm = get_ollama_llm()
    >>> print(llm.model)
    llama3
    >>> print(llm.base_url)
    http://localhost:11434
    >>> print(llm.metadata)
    context_window=3900 num_output=256 is_chat_model=True is_function_calling_model=True model_name='llama3' system_role=<MessageRole.SYSTEM: 'system'>
    """
    try:
        from llama_index.llms.ollama import Ollama
    except ImportError:
        raise ImportError(
            "Please install the `llama-index-llms-ollama` package to use the Ollama model."
        )

    return Ollama(
        model=model_id,
        base_url=base_url,
        temperature=temperature,
        context_window=context_window,
        request_timeout=request_timeout,
        prompt_key=prompt_key,
        json_mode=json_mode,
        additional_kwargs=additional_kwargs,
        is_function_calling_model=is_function_calling_model,
        keep_alive=keep_alive,
    )


def get_hugging_face_embedding(
    model_name: str = "BAAI/bge-small-en-v1.5", cache_folder: str = None
):
    """Get the hugging face embedding model.

    Parameters
    ----------
    model_name: str, optional, default is "BAAI/bge-small-en-v1.5"
        Name of the hugging face embedding model.
    cache_folder: str, optional, default is None
        Folder to cache the model. If not provided the function will search for
        - `LLAMA_INDEX_CACHE_DIR` in your environment variables.
        - `~/tmp/llama_index` if your OS is Linux.
        - `~/Library/Caches/llama_index` if your OS is MacOS.
        - `~/AppData/Local/llama_index` if your OS is Windows.

    Returns
    -------
    HuggingFaceEmbedding
        The hugging face embedding model.

    Raises
    ------
    ImportError
        If the `llama-index-embeddings-huggingface` package is not installed.

    Examples
    --------
    >>> from llama_utils.utils.models import get_hugging_face_embedding
    >>> embedding = get_hugging_face_embedding()
    >>> print(embedding.model_name)
    BAAI/bge-small-en-v1.5
    >>> print(embedding.max_length)
    512
    >>> print(embedding.embed_batch_size)
    10
    """
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError:
        raise ImportError(
            "Please install the `llama-index-embeddings-huggingface` package to use the Hugging Face embedding model."
        )

    embedding = HuggingFaceEmbedding(model_name=model_name, cache_folder=cache_folder)
    return embedding


def get_huggingface_llm(**kwargs):
    """Initializes and returns a HuggingFaceLLM instance with specified parameters.

    Parameters
    ----------
    kwargs: dict
        context_window : int, optional
            The maximum context window size, by default DEFAULT_CONTEXT_WINDOW.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate, by default DEFAULT_NUM_OUTPUTS.
        generate_kwargs : dict, optional
            Additional arguments for text generation, by default {"temperature": 0.75, "do_sample": False}.
        query_wrapper_prompt : str, optional
            The wrapper prompt for query execution, by default "Answer the following question succinctly and informatively.".
        tokenizer_name : str, optional
            The tokenizer model name, by default DEFAULT_HUGGINGFACE_MODEL.
        model_name : str, optional
            The model name, by default DEFAULT_HUGGINGFACE_MODEL.
        device_map : str, optional
            The device mapping strategy, by default "auto".
        tokenizer_kwargs : dict, optional
            Additional tokenizer arguments, by default {"max_length": 2048}.
        model_kwargs : dict, optional
            Additional model arguments, by default {"torch_dtype": torch.float16}.

    Returns
    -------
    HuggingFaceLLM
        An instance of the HuggingFaceLLM class.
    """
    try:
        import torch
        from llama_index.llms.huggingface import HuggingFaceLLM
    except ImportError:
        raise ImportError(
            "Please install the `llama-index-llms-huggingface` package to use the HuggingFaceLLM model."
        )

    return HuggingFaceLLM(
        context_window=kwargs.get("context_window", DEFAULT_CONTEXT_WINDOW),
        max_new_tokens=kwargs.get("max_new_tokens", DEFAULT_NUM_OUTPUTS),
        generate_kwargs=kwargs.get(
            "generate_kwargs", {"temperature": 0.75, "do_sample": False}
        ),
        query_wrapper_prompt=kwargs.get(
            "query_wrapper_prompt",
            "Answer the following question succinctly and informatively.",
        ),
        tokenizer_name=kwargs.get("tokenizer_name", DEFAULT_HUGGINGFACE_MODEL),
        model_name=kwargs.get("model_name", DEFAULT_HUGGINGFACE_MODEL),
        device_map=kwargs.get("device_map", "auto"),
        tokenizer_kwargs=kwargs.get("tokenizer_kwargs", {"max_length": 2048}),
        model_kwargs=kwargs.get(
            "model_kwargs",
            {"torch_dtype": torch.float16},
        ),
    )


class LLMModel:
    r"""Abstraction layer for different LLM providers: AzureOpenAI, Ollama, and HuggingFace.

    Parameters
    ----------
    model_type : str
        Type of the model ('azure', 'ollama', 'huggingface').
    **kwargs : dict
        Additional parameters for the model initialization.

    Examples
    --------
    Initialize an Azure OpenAI model:
        >>> from llama_utils.utils.models import LLMModel
        >>> from dotenv import load_dotenv
        >>> load_dotenv() # doctest: +SKIP
        >>> model = LLMModel(model_type='azure', model_id='gpt-4o', engine='4o') # doctest: +SKIP
        >>> print(model.base_model.model) # doctest: +SKIP
        gpt-4o
        >>> response = model.generate_response("Hello, how are you?") # doctest: +SKIP
        >>> print(response) # doctest: +SKIP
        Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?

    Initialize an Ollama model:
        >>> model = LLMModel(model_type='ollama', model_id='llama3.1')
        >>> response = model.generate_response("Hello, how are you?") # doctest: +SKIP
        >>> print(response) # doctest: +SKIP
        I'm just a language model, I don't have emotions or feelings like humans do, so I don't have good or bad days. However, I'm functioning properly and ready to help with any questions or tasks you may have! How about you? How's your day going?

    Initialize a HuggingFace model:
        >>> import os
        >>> cache_dir = os.getenv("CACHE_DIR")
        >>> model_kwargs = {}
        >>> model_kwargs["cache_dir"] = cache_dir
        >>> model_name = "distilgpt2"
        >>> model = LLMModel(
        ...     model_type='huggingface', model_name=model_name, tokenizer_name=model_name, model_kwargs=model_kwargs
        ... )
        >>> response = model.generate_response("Hello, how are you?") # doctest: +SKIP
        >>> print(response) # doctest: +SKIP
    """

    def __init__(self, model_type: str, **kwargs):
        """Initialize the LLM model."""
        self._model_type = model_type.lower()
        self._base_model = self._initialize_model(**kwargs)

    @property
    def base_model(self):
        """Get the base model."""
        return self._base_model

    @property
    def model_type(self):
        """Get the model type."""
        return self._model_type

    def _initialize_model(self, **kwargs):
        if self.model_type == "azure":
            return azure_open_ai(
                model_id=kwargs.get("model_id", "gpt-4o"),
                engine=kwargs.get("engine", "4o"),
            )
        elif self.model_type == "ollama":
            return get_ollama_llm(**kwargs)
        elif self.model_type == "huggingface":
            return get_huggingface_llm(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate_response(self, prompt: str, **kwargs):
        """Generate a response from the model.

        Parameters
        ----------
        prompt : str
            The input prompt.
        **kwargs : dict
            Additional parameters for generation.

        Returns
        -------
        str
            Generated response.

        Examples
        --------
        Generate response using Azure OpenAI:

        >>> model = LLMModel(model_type='azure', model_id='gpt-4o')
        >>> response = model.generate_response("What is AI?") # doctest: +SKIP
        >>> print(response) # doctest: +SKIP

        Generate response using Ollama:

        >>> model = LLMModel(model_type='ollama', model_id='llama3')
        >>> response = model.generate_response("Explain quantum mechanics.") # doctest: +SKIP
        >>> print(response) # doctest: +SKIP

        Generate response using HuggingFace:

        >>> model = LLMModel(model_type='huggingface', model_name='distilgpt2') # doctest: +SKIP
        >>> response = model.generate_response("Write a poem about the sea.") # doctest: +SKIP
        >>> print(response) # doctest: +SKIP
        """
        if self.model_type in ["azure", "ollama", "huggingface"]:
            return self.base_model.complete(prompt, **kwargs)
        else:
            raise ValueError("Invalid model type")
