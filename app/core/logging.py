from __future__ import annotations

import logging
import os
import sys


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL_ENV = "JUSTRAYZIST_LOG_LEVEL"
VERBOSE_LOGS_ENV = "JUSTRAYZIST_VERBOSE_LOGS"
NOISY_LIBRARY_LOGGERS = (
    "uvicorn.access",
    "httpx",
    "httpcore",
    "urllib3",
    "PIL",
)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _configure_library_loggers(verbose_logs: bool) -> None:
    library_level = logging.INFO if verbose_logs else logging.WARNING
    for logger_name in NOISY_LIBRARY_LOGGERS:
        logging.getLogger(logger_name).setLevel(library_level)


def configure_logging(level: str | None = None, *, verbose_logs: bool | None = None) -> None:
    effective_level = str(level or os.environ.get(LOG_LEVEL_ENV, "INFO")).strip() or "INFO"
    resolved_level = getattr(logging, effective_level.upper(), logging.INFO)
    os.environ[LOG_LEVEL_ENV] = logging.getLevelName(resolved_level)

    if verbose_logs is None:
        verbose_logs = _parse_bool_env(VERBOSE_LOGS_ENV, default=False)
    os.environ[VERBOSE_LOGS_ENV] = "1" if verbose_logs else "0"

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(resolved_level)
    else:
        logging.basicConfig(
            level=resolved_level,
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            stream=sys.stdout,
        )

    _configure_library_loggers(verbose_logs)
