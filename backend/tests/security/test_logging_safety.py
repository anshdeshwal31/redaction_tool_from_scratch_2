"""The logging facade must make it impossible to log document content (plan §14).

The benchmark adapter's oracle modes read document text from the files the benchmark injects: it must reach no
console output, no log and no file but the stage files that hold document text by design.
"""

from __future__ import annotations

import io
import json
import logging
import warnings
from pathlib import Path

import pytest
from test_bench_oracle import Bench, bench, letter
from test_bench_runner import RESIDUAL, InProcessPool

import mlredact.security.logging as facade
from inject_support import Docs, add_line, mention, token_polys
from mlredact.core.errors import ReasonCode
from mlredact.core.types import Stage
from mlredact.security.exceptions import sanitize
from mlredact.security.logging import UnsafeLogField, configure_logging, get_logger, scrub_terms
from mlredact.verify.checks import extract_all_text

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


# --------------------------------------------------------------------------- the benchmark's oracle modes
ORACLE_MODES = ("oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions")
# The stage files that hold document text by design.
TEXT_FILES = ["mentions.json", "pseudonyms.json", "text.json"]
# Verification finds a barcode it did not remove: oracle_text re-plans it, the modes that inject mentions leave it
# and hold the document back.
VERDICTS = {
    "oracle_text": {"released": True, "checks_failed": []},
    "oracle_mentions": {"released": False, "checks_failed": ["V4_RESIDUAL"]},
    "oracle_entities": {"released": False, "checks_failed": ["V4_RESIDUAL"]},
    "oracle_regions": {"released": False, "checks_failed": ["V4_RESIDUAL"]},
}


@pytest.fixture
def strict_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """MLREDACT_LOG_STRICT=1 (a log field that is not allowed fails the job instead of being dropped), every record
    kept."""
    monkeypatch.setenv("MLREDACT_LOG_STRICT", "1")
    monkeypatch.setitem(facade._state, "strict", True)  # the facade reads the variable when it is first imported
    caplog.set_level(logging.DEBUG)
    return caplog


def _with_canary() -> Docs:
    """The synthetic letter's injected files with "Copy to Dr <CANARY>" below it: CANARY is a mention, and the label
    of another one (on "to", too short a value to be searched for in the output)."""
    docs = letter()
    ids = add_line(docs, "lcanary", f"Copy to Dr {CANARY}", (60.0, 520.0))
    polys = token_polys(docs, ids)
    named = {"canary": (ids[3:], polys[3:]), "canary-label": (ids[1:2], polys[1:2])}
    docs["mentions.json"]["mentions"] += [
        mention("canary", "PERSON_NAME", CANARY, *named["canary"]),
        mention("canary-label", CANARY, None, *named["canary-label"]),
    ]
    docs["entities.json"]["clusters"].append({"id": "canary", "mentions": ["canary"], "attributes": {}})
    docs["plan.json"]["regions"] += [
        {"id": f"{m}-{k}", "page": 0, "poly": p, "action": "replace", "mentions": [m]}
        for m, (_ids, anchors) in named.items()
        for k, p in enumerate(anchors)
    ]
    return docs


def _shown(run: Bench, caplog: pytest.LogCaptureFixture) -> str:
    """The console output (the JSON log lines included) and every event mlredact logged, with its fields.  (Other
    libraries' records are written as a keyed hash of their text, if at all.)"""
    events = [
        f"{r.getMessage()} {getattr(r, 'mlredact_fields', '')}" for r in caplog.records if r.name.startswith("mlredact")
    ]
    return "\n".join([run.result.stdout, run.result.stderr, *events])


def _canary_in(text: str) -> bool:
    return any(word in text for word in CANARY.split())


@pytest.mark.parametrize("mode", ORACLE_MODES)
def test_injected_text_reaches_only_the_stage_files_that_hold_text(
    tmp_path: Path, strict_logs: pytest.LogCaptureFixture, mode: str
) -> None:
    run = bench(
        tmp_path,
        mode,
        _with_canary(),
        pool=InProcessPool(residuals=(RESIDUAL,)),
        options=["--log-level", "DEBUG"],
    )
    assert run.result.exit_code == 0, run.result.output
    # With strict logging a field that is not allowed would have failed the job (E_INTERNAL).
    assert run.json("run_info.json")["release"] == VERDICTS[mode]
    shown = _shown(run, strict_logs)
    assert '"event":"job.done"' in run.result.stderr and "job.done" in shown  # the logs were captured
    assert ("verify.replan" in shown) is (mode == "oracle_text")
    assert not _canary_in(shown)
    held = [p.name for p in sorted(run.out.iterdir()) if p.suffix == ".json" and _canary_in(p.read_text("utf-8"))]
    assert held == TEXT_FILES
    assert not _canary_in(extract_all_text(run.read("output.pdf")))


def _malformed(mode: str) -> Docs:
    """The injected files with CANARY, one file ``mode`` reads made invalid by it: a word with a 2-point polygon, a
    mention or a cluster with a key the schema does not have, a plan region with an action it does not have."""
    docs = _with_canary()
    if mode == "oracle_text":
        docs["text.json"]["pages"][0]["tokens"].append({"id": CANARY, "text": CANARY, "poly": [[0, 0], [1, 1]]})
    elif mode == "oracle_mentions":
        docs["mentions.json"]["mentions"][-1][CANARY] = CANARY
    elif mode == "oracle_entities":
        docs["entities.json"]["clusters"][-1][CANARY] = CANARY
    else:
        docs["plan.json"]["regions"][-1]["action"] = CANARY
    return docs


@pytest.mark.parametrize("mode", ORACLE_MODES)
def test_a_malformed_injected_file_shows_none_of_its_text(
    tmp_path: Path, strict_logs: pytest.LogCaptureFixture, mode: str
) -> None:
    run = bench(tmp_path, mode, _malformed(mode), options=["--log-level", "DEBUG"])
    assert run.result.exit_code == 1 and [p.name for p in run.out.iterdir()] == ["run_info.json"]
    info = run.read("run_info.json").decode("utf-8")
    assert json.loads(info)["release"] == {"released": False, "checks_failed": ["E_CONFIG_INVALID"]}
    assert not _canary_in(_shown(run, strict_logs) + info)
