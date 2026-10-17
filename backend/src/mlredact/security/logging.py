"""PII-safe structured logging.

Rules enforced here (see plan §14):

* mlredact code logs **events** with **allowlisted, typed fields** only — there is no free-text
  message argument at all.  String fields must match strict identifier patterns.
* Records from third-party libraries are reduced to ``logger``, ``level``, a keyed hash and the
  length of the message; their text is never written.
* Exception tracebacks are never formatted (their messages and source lines may contain document
  content); use :func:`mlredact.security.exceptions.sanitize` to log errors.
* A per-job scrub list (normalised removal-set values) is applied to every string field as a last
  line of defence.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import time
from collections.abc import Iterable, Iterator
from enum import Enum, StrEnum
from typing import Final, TextIO

from mlredact.core.errors import ReasonCode

_EVENT_RE: Final = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+){0,6}$")
_IDENT_RE: Final = re.compile(r"^[A-Za-z0-9_.:+\-]{1,96}$")
_HEX_RE: Final = re.compile(r"^[0-9a-f]{8,128}$")
_ID_RE: Final = re.compile(r"^[A-Za-z]{1,8}-[0-9a-f]{8,64}$")
_LOCATION_RE: Final = re.compile(r"^[A-Za-z0-9_.]{1,128}:[A-Za-z0-9_<>]{1,64}:[0-9]{1,6}$")

_INT = (int,)
_NUM = (int, float)
_BOOL = (bool,)

# field name -> (allowed python types, optional regex for strings)
_FIELDS: Final[dict[str, tuple[tuple[type, ...], re.Pattern[str] | None]]] = {
    # counters / sizes
    **{
        name: (_INT, None)
        for name in (
            "page",
            "pages",
            "count",
            "lines",
            "tokens",
            "candidates",
            "mentions",
            "entities",
            "ops",
            "regions",
            "bytes",
            "dpi",
            "width",
            "height",
            "workers",
            "worker",
            "attempt",
            "iteration",
            "dropped",
            "removed",
            "exit_code",
            "pid",
        )
    },
    # measurements
    **{name: (_NUM, None) for name in ("duration_ms", "ratio", "score", "confidence", "threshold")},
    # flags
    **{name: (_BOOL, None) for name in ("passed", "enabled", "cached", "strict")},
    # enums
    **{name: ((StrEnum,), None) for name in ("stage", "check", "category", "status", "action", "kind")},
    "reason": ((ReasonCode,), None),
    # identifier-like strings
    "job": ((str,), _ID_RE),
    "doc": ((str,), _HEX_RE),
    "input_sha256": ((str,), _HEX_RE),
    "config_hash": ((str,), _HEX_RE),
    "profile": ((str,), _IDENT_RE),
    "runtime_profile": ((str,), _IDENT_RE),
    "model": ((str,), _IDENT_RE),
    "detector": ((str,), _IDENT_RE),
    "engine": ((str,), _IDENT_RE),
    "version": ((str,), _IDENT_RE),
    "key_id": ((str,), _IDENT_RE),
    "error_type": ((str,), _IDENT_RE),
    "where": ((str,), _LOCATION_RE),
    "logger": ((str,), _IDENT_RE),
}

_SCRUB_TERMS: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "mlredact_scrub_terms", default=frozenset()
)
_PROCESS_HASH_KEY: Final = secrets.token_bytes(32)  # per-process: third-party hashes are not reversible


class UnsafeLogField(ValueError):
    """Raised in strict mode when code tries to log a field that is not allowlisted/valid."""


def _strict_default() -> bool:
    return os.environ.get("MLREDACT_LOG_STRICT", "0") == "1"


_state: dict[str, bool] = {"strict": _strict_default()}


def _scrub(value: str) -> str:
    terms = _SCRUB_TERMS.get()
    if terms:
        lowered = value.lower()
        if any(term in lowered for term in terms):
            return "<scrubbed>"
    return value


def _validate(name: str, value: object) -> object:
    spec = _FIELDS.get(name)
    if spec is None:
        raise UnsafeLogField(f"field {name!r} is not allowlisted")
    types, pattern = spec
    if value is None:
        return None
    if isinstance(value, bool) and bool not in types:
        raise UnsafeLogField(f"field {name!r} does not accept bool")
    if not isinstance(value, types):
        raise UnsafeLogField(f"field {name!r} has invalid type {type(value).__name__}")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        if pattern is None or not pattern.fullmatch(value):
            raise UnsafeLogField(f"field {name!r} value does not match its identifier pattern")
        return _scrub(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


class EventLogger:
    """Structured event logger.  ``log.info("ocr.page_done", page=3, tokens=120)``."""

    __slots__ = ("_logger",)

    def __init__(self, name: str) -> None:
        if not name.startswith("mlredact"):
            raise ValueError("EventLogger names must be in the mlredact namespace")
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, fields: dict[str, object]) -> None:
        if not self._logger.isEnabledFor(level):
            return
        if not _EVENT_RE.fullmatch(event):
            if _state["strict"]:
                raise UnsafeLogField("invalid event name")
            event = "invalid_event_name"
        clean: dict[str, object] = {}
        dropped = 0
        for key in sorted(fields):
            try:
                clean[key] = _validate(key, fields[key])
            except UnsafeLogField:
                if _state["strict"]:
                    raise
                dropped += 1
        if dropped:
            clean["dropped"] = dropped
        self._logger.log(level, event, extra={"mlredact_event": event, "mlredact_fields": clean})

    def debug(self, event: str, /, **fields: object) -> None:
        self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, /, **fields: object) -> None:
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, /, **fields: object) -> None:
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, /, **fields: object) -> None:
        self._emit(logging.ERROR, event, fields)


def get_logger(name: str) -> EventLogger:
    return EventLogger(name)


class _JsonFormatter(logging.Formatter):
    """Formats facade events; reduces any other record to a content-free summary."""

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name if _IDENT_RE.fullmatch(record.name) else "unknown",
        }
        event = getattr(record, "mlredact_event", None)
        if event is not None:
            base["event"] = event
            fields = getattr(record, "mlredact_fields", {})
            if isinstance(fields, dict):
                base.update(fields)
        else:
            # Third-party record: never write its text.  getMessage() may raise on bad args.
            try:
                message = record.getMessage()
            except Exception:
                message = "<unformattable>"
            digest = hashlib.blake2b(
                message.encode("utf-8", "replace"), key=_PROCESS_HASH_KEY, digest_size=8
            ).hexdigest()
            base.update({"event": "thirdparty.log", "msg_hash": digest, "msg_len": len(message)})
        # exc_info / stack_info are intentionally ignored: tracebacks can contain document content.
        return json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


_HANDLER_NAME: Final = "mlredact-json"


def configure_logging(level: str = "INFO", stream: TextIO | None = None, *, strict: bool | None = None) -> None:
    """Install the JSON handler on the root logger (idempotent)."""
    if strict is not None:
        _state["strict"] = strict
    logging.raiseExceptions = False  # never print record content on handler errors
    logging.captureWarnings(True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Chatty third-party loggers: keep warnings and above only (still content-free).
    for noisy in ("RapidOCR", "rapidocr", "PIL", "pikepdf", "onnxruntime", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextlib.contextmanager
def scrub_terms(terms: Iterable[str]) -> Iterator[None]:
    """Register job-specific sensitive values (lower-cased, len >= 3) for last-resort scrubbing."""
    normalised = frozenset(t.lower() for t in terms if len(t) >= 3)
    token = _SCRUB_TERMS.set(_SCRUB_TERMS.get() | normalised)
    try:
        yield
    finally:
        _SCRUB_TERMS.reset(token)
