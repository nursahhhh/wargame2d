from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

from infra.paths import STORAGE_DIR

# Simple, centralized logging setup for the backend.
DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s"
JSON_FORMAT = (
    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","line":%(lineno)d,"msg":"%(message)s"}'
)


def configure_logging(
    level: Union[str, int] = "INFO",
    *,
    json: bool = False,
    logfile: str | Path | None = STORAGE_DIR / "logs" / "backend.log",
) -> None:
    """
    Configure the root logger with stdout + optional file handler.

    Args:
        level: Logging level name or int (e.g., "DEBUG", logging.INFO).
        json: Emit JSON lines when True; otherwise a human-friendly format.
        logfile: File path to append logs; set to None to disable file output.
    """
    fmt = JSON_FORMAT if json else DEFAULT_FORMAT
    formatter = logging.Formatter(fmt)

    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    handlers.append(console)

    if logfile is not None:
        log_path = Path(logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger; configure_logging() should be called once on startup."""
    return logging.getLogger(name)


# Usage: from infra.logger import configure_logging, get_logger, STORAGE_DIR; configure_logging("DEBUG", logfile=STORAGE_DIR/"logs"/"backend.log"); log = get_logger(__name__); log.info("ready")
