from llama_utils.indexing.index_manager import IndexManager


def test_create_from_storage(storage_path: str):
    index_manager = IndexManager.create_from_storage(storage_path)
    assert isinstance(index_manager, IndexManager)
    assert len(index_manager.indexes) == 0
    assert index_manager.indexes == []
