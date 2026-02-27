from __future__ import annotations

import logging
import sys


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    if root.handlers:
        root.setLevel(resolved_level)
        return
    logging.basicConfig(
        level=resolved_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
    )
