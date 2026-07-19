"""Structured JSON-line logging.

Every event is one JSON object on stdout with a stable ``event`` name plus
whatever identifiers the caller attaches (``job_id``, ``step``, ``attempt``,
``worker_id``, ...), so a failed job can be reconstructed with ``grep job_id``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger("service")


class JsonLineFormatter(logging.Formatter):
    """Render each record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach the JSON formatter to the service logger (idempotent)."""
    if _logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLineFormatter())
    _logger.addHandler(handler)
    _logger.setLevel(level)
    _logger.propagate = False


def log_event(event: str, **fields: Any) -> None:
    """Emit one structured event."""
    _logger.info(event, extra={"event": event, "fields": fields})
