"""Structured logging: console, plain-text file and JSONL, keyed by run id.

Design
------
* Loggers live under the ``"lmfs"`` namespace and do **not** propagate to the
  root logger, so library or notebook logging configuration never duplicates
  our output and we never reconfigure theirs.
* Handlers installed by :func:`setup_logger` are tagged. Calling it again
  removes (and closes) the previously tagged handlers first, so re-running a
  notebook cell does not produce duplicate lines.
* The module knows nothing about training. Structured events are logged with
  :func:`log_event` and arbitrary JSON-serialisable fields.

JSONL record format (one object per line)::

    {"timestamp": "2026-11-10T14:03:22.123456+00:00", "level": "INFO",
     "logger": "lmfs", "run_id": "lstm/run_001", "message": "eval",
     "event": "eval", "fields": {"step": 500, "dev_nll": 2.1}}

Exceptions add an ``"exception"`` object with type, message and traceback.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

from utils.io import to_jsonable

ROOT_LOGGER_NAME = "lmfs"
_HANDLER_TAG = "_lmfs_handler"
_TEXT_FORMAT = "%(asctime)s | %(levelname)-7s | %(run_id)s | %(message)s"


class _RunIdFilter(logging.Filter):
    """Attach ``run_id`` to every record passing through a handler."""

    def __init__(self, run_id: str | None) -> None:
        super().__init__()
        self.run_id = run_id or "-"

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        return True


class _UTCFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        return _dt.datetime.fromtimestamp(record.created, tz=_dt.timezone.utc).isoformat(timespec="milliseconds")


class JsonlFormatter(logging.Formatter):
    """Format records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _dt.datetime.fromtimestamp(record.created, tz=_dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        fields = getattr(record, "fields", None)
        if fields:
            try:
                payload["fields"] = to_jsonable(fields)
            except TypeError as exc:
                payload["fields"] = {"_unserialisable": repr(fields), "_error": str(exc)}
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value),
                "traceback": self.formatException(record.exc_info),
            }
        return json.dumps(payload, allow_nan=False, default=str)


def _remove_tagged_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        if getattr(h, _HANDLER_TAG, False):
            logger.removeHandler(h)
            h.close()


def _tag(handler: logging.Handler, run_id: str | None, formatter: logging.Formatter) -> logging.Handler:
    setattr(handler, _HANDLER_TAG, True)
    handler.addFilter(_RunIdFilter(run_id))
    handler.setFormatter(formatter)
    return handler


def setup_logger(
    name: str = ROOT_LOGGER_NAME,
    *,
    run_id: str | None = None,
    log_dir: str | Path | None = None,
    level: int | str = logging.INFO,
    console: bool = True,
    text_file: bool = True,
    jsonl: bool = True,
) -> logging.Logger:
    """Configure and return a logger. Safe to call repeatedly.

    With ``log_dir`` set, writes ``run.log`` (human-readable) and
    ``events.jsonl`` (structured) inside it. ``name`` must be ``"lmfs"`` or a
    child such as ``"lmfs.data"``.
    """
    if name != ROOT_LOGGER_NAME and not name.startswith(ROOT_LOGGER_NAME + "."):
        raise ValueError(f"logger name must be '{ROOT_LOGGER_NAME}' or start with '{ROOT_LOGGER_NAME}.'")
    if isinstance(level, str):
        numeric = logging.getLevelName(level.upper())
        if not isinstance(numeric, int):
            raise ValueError(f"unknown log level {level!r}")
        level = numeric
    logger = logging.getLogger(name)
    _remove_tagged_handlers(logger)
    logger.setLevel(level)
    logger.propagate = False

    text_fmt = _UTCFormatter(_TEXT_FORMAT)
    if console:
        logger.addHandler(_tag(logging.StreamHandler(sys.stderr), run_id, text_fmt))
    if log_dir is not None and (text_file or jsonl):
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if text_file:
            logger.addHandler(_tag(logging.FileHandler(directory / "run.log", encoding="utf-8"), run_id, text_fmt))
        if jsonl:
            logger.addHandler(
                _tag(logging.FileHandler(directory / "events.jsonl", encoding="utf-8"), run_id, JsonlFormatter())
            )
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Remove and close the handlers installed by :func:`setup_logger`."""
    _remove_tagged_handlers(logger)


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Log a structured event. ``fields`` appear under ``"fields"`` in JSONL
    and as ``key=value`` pairs in text output."""
    text = event if not fields else f"{event} " + " ".join(f"{k}={_short(v)}" for k, v in fields.items())
    logger.log(level, text, extra={"event": event, "fields": fields})


def log_exception(logger: logging.Logger, message: str, exc: BaseException | None = None, **fields: Any) -> None:
    """Log an exception with traceback. Inside ``except`` the active exception
    is used; otherwise pass ``exc`` explicitly. Never suppresses anything."""
    exc_info: Any = (type(exc), exc, exc.__traceback__) if exc is not None else True
    logger.error(message, exc_info=exc_info, extra={"event": "exception", "fields": fields})


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
