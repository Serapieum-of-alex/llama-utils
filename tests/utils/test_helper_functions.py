from llama_utils.utils.helper_functions import generate_content_hash, is_sha256


def test_generate_content_hash():
    content = "This is a test document"
    assert (
        generate_content_hash(content)
        == "c41cbbf2c21619e1d51dd729dbd9dd73693672ac0e358bfcda467827ba41bdf7"
    )


def test_is_sha256():
    hash_string = "b94d27b9934d3e08a52e52d7da7dabfade34ebf2d9a1e6f1cb7fd8d3cb3a53f7"
    assert is_sha256(hash_string)
    assert not is_sha256("b94d27b9934d3e08a52e52d7da7dabfade34e7")
