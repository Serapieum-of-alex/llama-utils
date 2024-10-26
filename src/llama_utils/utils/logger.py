import sys
import logging


class Logger:
    def __init__(self, name: str, level: int = logging.INFO, file_name: str = None):
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=file_name if not None else "llama-utils.log",
        )
        self.logger = logging.getLogger(name)
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
