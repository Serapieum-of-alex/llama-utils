import hashlib
import re


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


def is_sha256(string: str) -> bool:
    """Check if a string is a valid SHA-256 hash.

    Parameters
    ----------
    string: str
        The string to check.

    Returns
    -------
    bool
        True if the string is a valid SHA-256 hash, False otherwise.
    """
    # SHA-256 hash must be 64 characters long and contain only hexadecimal characters
    return bool(re.fullmatch(r"[a-fA-F0-9]{64}", string))
