from llama_utils.retrieval.vector_store import VectorStore


def test_create_storage_context():
    # vector_store = VectorStore()
    storage_context = VectorStore._create_simple_storage_context()
    assert storage_context is not None, "Storage context not created."
    assert list(storage_context.vector_stores.keys()) == ["default", "image"]
