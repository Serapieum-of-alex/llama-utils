from llama_utils.indexing.index_manager import IndexManager


def test_create_from_empty_storage(storage_path: str):
    index_manager = IndexManager.create_from_storage(storage_path)
    assert isinstance(index_manager, IndexManager)
    assert len(index_manager.indexes) == 0
    assert index_manager.indexes == []


def test_create_from_storage(paul_grahm_essay_storage: str):
    index_manager = IndexManager.create_from_storage(paul_grahm_essay_storage)
    assert isinstance(index_manager, IndexManager)
    assert len(index_manager.indexes) == 2
    assert index_manager.ids == [
        "8d57e294-fd17-43c9-9dec-a12aa7ea0751",
        "edd0d507-9100-4cfb-8002-2267449c6668",
    ]
