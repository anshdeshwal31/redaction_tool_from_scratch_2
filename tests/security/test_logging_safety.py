"""The logging facade must make it impossible to log document content (plan §14)."""

from __future__ import annotations

import io
import json
import logging
import warnings

import pytest

from mlredact.core.errors import ReasonCode
from mlredact.core.types import Stage
from mlredact.security.exceptions import sanitize
from mlredact.security.logging import UnsafeLogField, configure_logging, get_logger, scrub_terms

CANARY = "Zelphinia Quarrington"  # a canary value that must never reach any log output


@pytest.fixture
def log_stream() -> io.StringIO:
    stream = io.StringIO()
    configure_logging("DEBUG", stream, strict=True)
    return stream


def _records(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_allowlisted_fields_are_emitted(log_stream: io.StringIO) -> None:
    get_logger("mlredact.test").info("ocr.page_done", page=3, tokens=120, stage=Stage.OCR, duration_ms=12.5)
    (rec,) = _records(log_stream)
    assert rec["event"] == "ocr.page_done" and rec["page"] == 3 and rec["stage"] == "ocr"


def test_unknown_field_is_rejected_in_strict_mode(log_stream: io.StringIO) -> None:
    with pytest.raises(UnsafeLogField):
        get_logger("mlredact.test").info("x.y", text=CANARY)
    assert CANARY not in log_stream.getvalue()


def test_identifier_fields_reject_free_text(log_stream: io.StringIO) -> None:
    with pytest.raises(UnsafeLogField):
        get_logger("mlredact.test").info("x.y", model=CANARY)  # contains a space


def test_lenient_mode_drops_unsafe_fields(log_stream: io.StringIO) -> None:
    configure_logging("DEBUG", log_stream, strict=False)
    get_logger("mlredact.test").info("x.y", text=CANARY, page=1)
    out = log_stream.getvalue()
    assert CANARY not in out
    (rec,) = _records(log_stream)
    assert rec["dropped"] == 1 and rec["page"] == 1


def test_third_party_messages_are_never_written(log_stream: io.StringIO) -> None:
    logging.getLogger("somelib").error("failed on %s", CANARY)
    logging.getLogger("somelib").error("failed", exc_info=ValueError(CANARY))
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(f"bad value {CANARY}", stacklevel=1)
    out = log_stream.getvalue()
    assert CANARY not in out
    assert all(r["event"] == "thirdparty.log" for r in _records(log_stream))


def test_scrub_terms_catch_identifier_shaped_values(log_stream: io.StringIO) -> None:
    with scrub_terms(["quarrington"]):
        get_logger("mlredact.test").info("x.y", model="model-quarrington-v1")
    assert "quarrington" not in log_stream.getvalue().lower()


def test_sanitize_drops_third_party_messages() -> None:
    try:
        int(CANARY)
    except ValueError as exc:
        info = sanitize(exc)
    assert info.code is ReasonCode.INTERNAL
    assert CANARY not in repr(info) and CANARY not in str(info.as_fields())
    assert info.error_type == "builtins.ValueError"
