import hashlib


class HelperFunctions:
    """A collection of helper functions used across different modules (e.g., text preprocessing, validation)."""

    def __init__(self):
        pass


def generate_content_hash(content: str):
    """Generate a hash for the document content using SHA-256.

    Parameters
    ----------
    content: str
        The content of the document.

    Returns
    -------
    str
        The SHA-256 hash of the content.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
