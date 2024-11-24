"""Logger."""

import logging
import sys


class Logger:
    """Logger class."""

    def __init__(self, name: str, level: int = logging.INFO, file_name: str = None):
        """Initialize the logger."""
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=file_name if not None else "llama-utils.log",
        )
        self.logger = logging.getLogger(name)
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
