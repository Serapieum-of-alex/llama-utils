from llama_utils.utils.helper_functions import generate_content_hash


def test_generate_content_hash():
    content = "This is a test document"
    assert (
        generate_content_hash(content)
        == "c41cbbf2c21619e1d51dd729dbd9dd73693672ac0e358bfcda467827ba41bdf7"
    )
