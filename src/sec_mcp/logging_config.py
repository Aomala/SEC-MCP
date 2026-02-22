"""Centralized logging configuration.

Call setup_logging() at application startup (server.py, chat_app.py) to
configure structured JSON logging for production or readable output for dev.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal


class JSONFormatter(logging.Formatter):
    """Simple JSON log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include any extra fields
        for key in ("request_id", "ticker", "tool", "duration_ms"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def setup_logging(
    level: str = "INFO",
    fmt: Literal["json", "text"] = "text",
) -> None:
    """Configure root logger for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        fmt: Output format — "json" for production, "text" for development
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)

    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))

    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
