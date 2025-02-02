"""LLMs and embedding models."""

import os
from typing import Any, Dict, Optional, Union
from warnings import warn


DEFAULT_CONTEXT_WINDOW = 3900


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
    model_name: str = "BAAI/bge-base-en-v1.5", cache_folder: str = None
):
    """Get the hugging face embedding model.

    Parameters
    ----------
    model_name: str, optional, default is "BAAI/bge-base-en-v1.5"
        Name of the hugging face embedding model.
    cache_folder: str, optional, default is None
        Folder to cache the model.

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
    BAAI/bge-base-en-v1.5
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
