"""The real runner behind ``mlredact bench``: the job trace, and the hand-over of output the tool does not release.

``Redactor.run`` runs as in production (intake, redaction, PDF build, checks V1-V6, re-plans), except that the
worker tasks run in this process and page analysis reads the PDF's own text layer instead of running OCR, so no
models are needed.  The letter is synthetic (``tools/synthdocs``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import pytest
import yaml
from synthdocs import make_ime_letter
from typer.testing import CliRunner

import mlredact.bench.environment as bench_env
import mlredact.pipeline.runner as runner_module
import mlredact.pipeline.workers as workers_module
from mlredact.cli.main import app
from mlredact.config.loader import load_config
from mlredact.config.schema import AppConfig
from mlredact.core.canonical import canonical_json
from mlredact.core.errors import ProcessingError, ReasonCode
from mlredact.core.geometry import BBox
from mlredact.core.model import PageAnalysis, Region, RenderOp
from mlredact.core.types import JobStatus, RegionKind
from mlredact.ingest.render import render_page
from mlredact.ocr.fusion import fuse_embedded
from mlredact.pipeline.runner import JobResult, Redactor
from mlredact.pipeline.shared import BlobRef
from mlredact.pipeline.workers import RedactedPage

LETTER = make_ime_letter(scanned=False).pdf
# The client profile without models (no NER, no OCR engine B), at the lowest resolution to keep the runs fast.
NO_MODELS: dict[str, Any] = {
    "render": {"dpi": 150},
    "ocr": {"engine_b": {"enabled": False}},
    "detection": {"ner": {"gliner": {"enabled": False}, "privacy_filter": {"enabled": False}}},
}
# The letter's text ends 912 px down its 1754 px page (150 dpi): these boxes lie below it.
SIGNATURE = Region("G-signature", 0, RegionKind.SIGNATURE, BBox(100, 1500, 400, 1600), 0.9)
RESIDUAL = BBox(1000, 1600, 1100, 1700)
ANOTHER_RESIDUAL = BBox(700, 1600, 800, 1700)


@dataclass
class Build:
    """One output the runner built: the operations it was drawn from, its redacted pages and the PDF."""

    ops: dict[int, tuple[RenderOp, ...]]
    pages: list[RedactedPage]
    pdf: bytes | None = None

    @property
    def erased(self) -> dict[tuple[str, BBox], BBox]:
        return {(a.ref_id, a.planned): a.erased for page in self.pages for a in page.applied}


@dataclass
class InProcessPool:
    """Stands in for ``WorkerPool``: the real worker tasks, run in this process, except page analysis, which
    reads the PDF's text layer (and finds a signature) instead of running OCR.

    The verification re-read of an output finds a barcode outside every planned box at ``residuals[k]`` on its
    k-th pass (each one forces a re-plan, up to ``verification.max_replans``); re-read ``fail_reread`` and page
    redaction ``fail_redaction`` (counted from 1) fail, and with ``fail_analysis`` so does the first reading.
    ``builds`` records every output built."""

    workers: ClassVar[int] = 1  # for the run record

    residuals: tuple[BBox, ...] = ()
    fail_reread: int = 0
    fail_redaction: int = 0
    fail_analysis: bool = False
    rereads: int = 0
    redactions: int = 0
    builds: list[Build] = field(default_factory=list)

    def install(self, mp: pytest.MonkeyPatch) -> None:
        """Every ``Redactor`` made from now on uses this pool, and reads no model files."""
        build_pdf = runner_module.build_pdf

        def recorded_build(*args: Any, **kwargs: Any) -> bytes:
            pdf = build_pdf(*args, **kwargs)
            self.builds[-1].pdf = pdf
            return pdf

        mp.setattr(workers_module, "_STATE", {})
        mp.setattr(runner_module, "WorkerPool", self.start)
        mp.setattr(runner_module, "resolve", lambda *_args: {})
        mp.setattr(runner_module, "build_pdf", recorded_build)
        mp.delenv("MLREDACT_SURROGATE_KEY", raising=False)

    def start(self, config: AppConfig, *_args: Any, **_kwargs: Any) -> InProcessPool:
        """``WorkerPool(config, ...)``: this process becomes the worker (what ``init_worker`` does)."""
        workers_module._STATE["config"] = config
        return self

    def reset(self) -> None:
        self.rereads = self.redactions = 0
        self.builds = []

    def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        return self.map(fn, [args])[0]

    def map(self, fn: Callable[..., Any], arg_list: Sequence[tuple[Any, ...]]) -> list[Any]:
        if fn is workers_module.task_analyse:
            return [self._analyse(*args) for args in arg_list]
        if fn is workers_module.task_redact:
            self.redactions += 1
            if self.redactions == self.fail_redaction:
                raise ProcessingError(ReasonCode.RENDER_FAILED, page=0)
            pages = [fn(*args) for args in arg_list]
            self.builds.append(Build({page: ops for _ref, page, ops, _notice in arg_list if ops}, pages))
            return pages
        return [fn(*args) for args in arg_list]

    def close(self) -> None:
        cached = workers_module._STATE.pop("doc", None)
        if cached is not None:
            cached[1].close()

    def _analyse(self, ref: BlobRef, page: int, profile: str = "primary") -> PageAnalysis:
        regions: tuple[Region, ...] = (SIGNATURE,)
        if profile == "verify":
            self.rereads += 1
            if self.rereads == self.fail_reread:
                raise ProcessingError(ReasonCode.OCR_FAILED, page=page)
            found = self.residuals[self.rereads - 1 : self.rereads]
            regions = tuple(Region(f"G-residual-{self.rereads}", page, RegionKind.BARCODE, b, 1.0) for b in found)
        elif self.fail_analysis:
            raise ProcessingError(ReasonCode.OCR_FAILED, page=page)
        cfg = workers_module._config()
        rendered = render_page(workers_module._document(ref), page, cfg.render, cfg.limits, with_words=True)
        lines = fuse_embedded([], rendered.words, rendered.rgb, page)
        kind = workers_module._page_kind(rendered.embedded_char_count, rendered.image_object_count)
        return PageAnalysis(rendered.geometry, kind, lines, rendered.embedded_char_count, regions)


class Case(NamedTuple):
    pool: dict[str, Any]  # InProcessPool settings
    status: JobStatus
    reasons: list[str]
    builds: int  # outputs built
    replanned: bool  # the last output built carries a re-plan's operations


CASES = {
    "released": Case({}, JobStatus.RELEASED, [], 1, False),
    "held back after a re-plan": Case(
        {"residuals": (RESIDUAL, ANOTHER_RESIDUAL)}, JobStatus.QUARANTINED, ["V4_RESIDUAL"], 2, True
    ),
    "re-read failed": Case({"fail_reread": 1}, JobStatus.QUARANTINED, ["E_OCR_FAILED"], 1, False),
    # The re-plan is planned but never drawn: the job's last plan is newer than its last output.
    "re-planned redaction failed": Case(
        {"residuals": (RESIDUAL,), "fail_redaction": 2}, JobStatus.QUARANTINED, ["E_RENDER_FAILED"], 1, False
    ),
    "nothing built": Case({"fail_analysis": True}, JobStatus.QUARANTINED, ["E_OCR_FAILED"], 0, False),
}


class Job(NamedTuple):
    plain: JobResult  # run without a trace
    traced: JobResult  # run with one
    builds: list[Build]  # the outputs the traced run built


class Jobs:
    """Each case run once without and once with a trace, on first use (every run renders the page)."""

    def __init__(self) -> None:
        self._done: dict[str, Job] = {}

    def __getitem__(self, name: str) -> Job:
        if name not in self._done:
            pool = InProcessPool(**CASES[name].pool)
            config = load_config("client", (), {**NO_MODELS, "surrogate": {"allow_development_key": True}})
            with pytest.MonkeyPatch.context() as mp:
                pool.install(mp)
                with Redactor(config) as redactor:
                    plain = redactor.run(LETTER, keep_held_back=True)
                    pool.reset()
                    traced = redactor.run(LETTER, keep_held_back=True, trace=True)
            self._done[name] = Job(plain, traced, pool.builds)
        return self._done[name]


@pytest.fixture(scope="module")
def jobs() -> Jobs:
    return Jobs()


def _run_record(result: JobResult) -> dict[str, Any]:
    return {k: v for k, v in result.run_record.items() if k not in ("job_id", "timings_ms")}


# ------------------------------------------------------------------------------- the runner
@pytest.mark.parametrize("name", CASES)
def test_a_trace_changes_nothing_else_in_the_result(jobs: Jobs, name: str) -> None:
    plain, traced = jobs[name].plain, jobs[name].traced
    assert (traced.status, traced.manifest["reasons"]) == (CASES[name].status, CASES[name].reasons)
    assert plain.trace is None and traced.trace is not None
    assert (plain.status, plain.pdf, plain.held_back_pdf) == (traced.status, traced.pdf, traced.held_back_pdf)
    assert canonical_json(plain.manifest) == canonical_json(traced.manifest)
    assert _run_record(plain) == _run_record(traced)


@pytest.mark.parametrize("name", CASES)
def test_the_trace_holds_the_last_output_built_and_what_it_was_drawn_from(jobs: Jobs, name: str) -> None:
    case, job = CASES[name], jobs[name]
    trace = job.traced.trace
    assert trace is not None and len(job.builds) == case.builds
    if not job.builds:
        assert trace.pdf is None and not trace.ops  # bench then has no output to hand over
        return
    last = job.builds[-1]
    assert trace.pdf is not None and trace.pdf == last.pdf
    assert dict(trace.ops) == last.ops and dict(trace.erased) == last.erased
    assert any(op.ref_id.startswith("R-") for ops in trace.ops.values() for op in ops) is case.replanned
    assert job.traced.pdf == (trace.pdf if case.status is JobStatus.RELEASED else None)


def test_run_keeps_only_what_verification_held_back(jobs: Jobs) -> None:
    """``keep_held_back`` (``run --keep-held-back``) keeps output only when the checks held it back; bench
    hands over any output that was built."""
    held, failed = jobs["held back after a re-plan"].traced, jobs["re-read failed"].traced
    assert held.trace is not None and failed.trace is not None
    assert held.pdf is None and held.held_back_pdf == held.trace.pdf
    assert failed.held_back_pdf is None and failed.trace.pdf is not None


# ------------------------------------------------------------------------------ the command
@pytest.mark.parametrize(
    ("name", "note"),
    [
        ("held back after a re-plan", "held back by the tool's own verification"),
        ("re-read failed", "a step failed after the output was built (E_OCR_FAILED)"),
        ("nothing built", "the pipeline built no output"),
    ],
)
def test_bench_hands_over_the_last_output_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, note: str
) -> None:
    case = CASES[name]
    pool = InProcessPool(**case.pool)
    pool.install(monkeypatch)
    monkeypatch.setattr(bench_env, "candidate_version", lambda root=None: "0.1.0+gtest")
    monkeypatch.delenv("MLREDACT_CONFIG", raising=False)
    monkeypatch.setenv("MLREDACT_MODELS_DIR", str(tmp_path))  # never read: model files are not resolved
    monkeypatch.chdir(tmp_path)
    config, src, out = tmp_path / "no-models.yaml", tmp_path / "input.pdf", tmp_path / "out"
    config.write_text(yaml.safe_dump(NO_MODELS), "utf-8")
    src.write_bytes(LETTER)
    res = CliRunner().invoke(app, ["bench", "--input", str(src), "--workdir", str(out), "--config", str(config)])
    assert res.exit_code == (0 if case.builds else 1), res.output
    info = json.loads((out / "run_info.json").read_text("utf-8"))
    assert info["release"] == {"released": False, "checks_failed": case.reasons}
    assert note in info["notes"]
    assert ("verification" in info["notes"]) is (name == "held back after a re-plan")
    if not pool.builds:
        assert not (out / "output.pdf").exists()
        return
    last = pool.builds[-1]
    assert (out / "output.pdf").read_bytes() == last.pdf
    # plan.json describes the operations output.pdf was drawn from, a re-plan's included.
    plan = json.loads((out / "plan.json").read_text("utf-8"))["regions"]
    assert {r["id"].rsplit(".", 1)[0] for r in plan} == {op.ref_id for ops in last.ops.values() for op in ops}
    assert any(r["id"].startswith("R-") for r in plan) is case.replanned
