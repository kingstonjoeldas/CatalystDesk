"""Logging utilities with request ID tracking."""

import logging
import sys
from typing import Optional
import uuid
from pathlib import Path


# Thread-local storage for request IDs
import threading
_request_id_local = threading.local()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set or generate a request ID for this thread."""
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    _request_id_local.request_id = request_id
    return request_id


def get_request_id() -> str:
    """Get the current request ID, or generate one if not set."""
    if not hasattr(_request_id_local, "request_id"):
        return set_request_id()
    return _request_id_local.request_id


class RequestIDFormatter(logging.Formatter):
    """Formatter that includes request ID in all log lines."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = get_request_id()
        record.request_id = request_id
        return super().format(record)


def get_request_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Create a logger with request ID tracking.

    Args:
        name: Logger name (typically __name__)
        log_dir: Directory for log file. If None, logs only to stderr.

    Returns:
        logging.Logger configured with request ID formatter.
    """
    from .config import Config

    logger = logging.getLogger(name)

    # Skip if already configured
    if logger.handlers:
        return logger

    logger.setLevel(Config.LOG_LEVEL)

    # Formatter: include request ID
    fmt = "%(asctime)s | %(request_id)s | %(name)s | %(levelname)s | %(message)s"
    formatter = RequestIDFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(Config.LOG_LEVEL)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (if log_dir provided)
    log_dir = log_dir or Config.LOG_DIR
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "catalyst.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(Config.LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Module-level logger
logger = get_request_logger(__name__)
