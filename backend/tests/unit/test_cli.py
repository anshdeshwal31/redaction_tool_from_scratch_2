"""CLI output files for released and quarantined jobs (the pipeline itself is stubbed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

import mlredact.pipeline.runner as runner_module
from mlredact.cli.main import HELD_BACK_PDF, app
from mlredact.core.types import JobStatus
from mlredact.pipeline.runner import JobResult

MANIFEST = {"reasons": ["V2_TEXT_LEAK"], "job": {"input_sha256": "0" * 64}}


class _StubRedactor:
    calls: ClassVar[list[dict[str, Any]]] = []
    status: ClassVar[JobStatus] = JobStatus.QUARANTINED

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def __enter__(self) -> _StubRedactor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, data: bytes, **kwargs: Any) -> JobResult:
        self.calls.append(kwargs)
        if self.status is JobStatus.RELEASED:
            return JobResult(JobStatus.RELEASED, b"%PDF-released", {"reasons": []}, {})
        held = b"%PDF-held" if kwargs["keep_held_back"] else None
        return JobResult(JobStatus.QUARANTINED, None, MANIFEST, {}, held_back_pdf=held)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[_StubRedactor]:
    _StubRedactor.calls = []
    _StubRedactor.status = JobStatus.QUARANTINED
    monkeypatch.setattr(runner_module, "Redactor", _StubRedactor)
    return _StubRedactor


def _run(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    src = tmp_path / "input.pdf"
    src.write_bytes(b"%PDF-1.7 stub")
    out = tmp_path / "out"
    res = CliRunner().invoke(app, ["run", str(src), "--out", str(out), "--dev-key", *extra])
    return res.exit_code, out


def test_quarantined_job_keeps_no_redacted_copy_by_default(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    code, out = _run(tmp_path)
    assert code == 2
    assert stub.calls[0]["keep_held_back"] is False
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "quarantine.json", "run.json"]


def test_keep_held_back_writes_the_copy_under_a_name_that_is_not_a_release(
    tmp_path: Path, stub: type[_StubRedactor]
) -> None:
    code, out = _run(tmp_path, "--keep-held-back")
    assert code == 2
    assert (out / HELD_BACK_PDF).read_bytes() == b"%PDF-held"
    assert not (out / "redacted.pdf").exists()


def test_keep_held_back_changes_nothing_for_a_released_job(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    stub.status = JobStatus.RELEASED
    code, out = _run(tmp_path, "--keep-held-back")
    assert code == 0
    assert (out / "redacted.pdf").read_bytes() == b"%PDF-released"
    assert not (out / HELD_BACK_PDF).exists()
