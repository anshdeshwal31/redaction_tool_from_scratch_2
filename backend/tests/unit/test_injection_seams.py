"""The runner's seams for injected stage inputs (``Redactor.run(inject=...)``, the benchmark's oracle modes), driven
by a stand-in for the benchmark adapter's implementation: what each hook returns is what the job uses, a hook that
declines leaves tool 2's own stage running, and without an injection nothing changes.

Jobs run as in tests/unit/test_bench_runner.py (its in-process worker pool, no models) on its synthetic letter.  All
text is fictitious.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from test_bench_runner import LETTER, NO_MODELS, RESIDUAL, InProcessPool

import mlredact.pipeline.workers as workers_module
from mlredact.config.loader import load_config
from mlredact.config.schema import AppConfig
from mlredact.core.canonical import content_id
from mlredact.core.errors import ProcessingError, ReasonCode
from mlredact.core.geometry import BBox, union_all
from mlredact.core.model import Evidence, Line, Mention, PageAnalysis, PageGeometry, Region, RenderOp, Token
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength, JobStatus, RegionKind
from mlredact.detect.engine import DetectionResult
from mlredact.layout.regions import PageRegions
from mlredact.pipeline.injection import (
    INJECTED_ENGINE,
    InjectedPage,
    InjectedStage,
    InjectedToken,
    InjectReport,
    injected_lines,
)
from mlredact.pipeline.runner import JobResult, Redactor, RegionDecision
from mlredact.pipeline.shared import SharedBlob
from mlredact.policy.engine import Decision
from mlredact.resolve.persons import resolve_persons
from mlredact.surrogate.assign import MentionSurrogate
from mlredact.text.views import build_line_view
from mlredact.verify.residual import ResidualHit

ROOT = Path(__file__).resolve().parents[2]
# What production must never load: the benchmark adapter and the injected stage inputs.
INJECTION_MODULES = ("mlredact.bench", "mlredact.pipeline.injection")
CONFIG = load_config("client", (), {**NO_MODELS, "surrogate": {"allow_development_key": True}})
# A signature the injected mentions become (below the injected words, away from every other box).
SIGNATURE = Region("G-injected", 0, RegionKind.SIGNATURE, BBox(150, 1100, 450, 1180), 1.0, "inject", "inject.SIGNATURE")


def _page(line: str) -> InjectedPage:
    """One injected line in the letter's blank lower half (display points; its text ends 438 pt down the page)."""
    tokens = []
    x = 60.0
    for i, word in enumerate(line.split()):
        width = 5.5 * len(word)
        tokens.append(InjectedToken(0, i, ((x, 500.0), (x + width, 500.0), (x + width, 511.0), (x, 511.0)), 1.0, word))
        x += width + 3.0
    return InjectedPage(0, tuple(tokens))


PERSON = _page("Mr Zelda Quimby attended.")
# Mostly not PII, so tool 2 replaces the number alone instead of re-typesetting the line.
PHONE = _page("Ring the clinic on 0412 345 678 before noon")
A4 = PageGeometry(0, 595.276, 841.89, 150, 1240, 1754)


@dataclass
class FakeInputs:
    """Stands in for the benchmark adapter's stage inputs: answers every hook from its settings and records what the
    runner handed it.  ``stages`` and ``plans``: replace detection and policy, and planning (otherwise the hooks
    decline); ``replans``: let verification re-plan the hits it found."""

    page: InjectedPage
    stages: bool = True
    plans: bool = True
    replans: bool = False
    detect_regions: bool = False
    calls: list[str] = field(default_factory=list)
    staged: InjectedStage | None = None
    planned: dict[int, tuple[RenderOp, ...]] | None = None
    plan_args: dict[str, Any] = field(default_factory=dict)
    hits: list[list[ResidualHit]] = field(default_factory=list)
    reported: InjectReport = field(
        default_factory=lambda: InjectReport("oracle_test", "text_layer", notes=("1 mention",))
    )

    def pages(self, page_count: int) -> list[InjectedPage]:
        self.calls.append("pages")
        return [self.page if i == self.page.index else InjectedPage(i) for i in range(page_count)]

    def stage(self, pages: Sequence[PageAnalysis], config: AppConfig, doc_key: str) -> InjectedStage | None:
        self.calls.append("stage")
        if not self.stages:
            return None
        ids = tuple(t.token_id for t in list(pages[0].tokens())[1:3])  # "Zelda Quimby"
        evidence = (Evidence("bench.inject", "1", "inject.PERSON_NAME", EvidenceStrength.STRONG, 1.0),)
        mention = Mention(
            content_id("M", doc_key, ids[0], ids[-1], "person"), EntityType.PERSON, ids, evidence, "Zelda Quimby"
        )
        view = build_line_view(pages)
        detectors = (("bench.inject", "1"), ("resolve.persons", "1"))
        detection = DetectionResult((mention,), (), detectors, 0, resolve_persons([mention], view), view)
        with_region = tuple(replace(p, regions=(SIGNATURE,)) if p.geometry.index == 0 else p for p in pages)
        self.staged = InjectedStage(with_region, detection, (Decision(mention, ActionKind.SURROGATE),))
        return self.staged

    def plan(
        self,
        *,
        decisions: Sequence[Decision],
        surrogates: Mapping[str, MentionSurrogate],
        tokens: Mapping[str, Token],
        pages: Sequence[PageAnalysis],
        region_actions: Mapping[str, ActionKind],
        rotated: AbstractSet[str],
        config: AppConfig,
    ) -> dict[int, tuple[RenderOp, ...]] | None:
        self.calls.append("plan")
        self.plan_args = {
            "decisions": tuple(decisions),
            "surrogates": dict(surrogates),
            "tokens": dict(tokens),
            "pages": tuple(pages),
            "region_actions": dict(region_actions),
            "rotated": rotated,
            "config": config,
        }
        if not self.plans:
            return None
        # Unpadded, and nothing for the signature: tool 2's planning would pad, and synthesise a signature.
        ops = []
        for d in decisions:
            box = union_all([tokens[t].bbox for t in d.mention.token_ids])
            ops.append(RenderOp(0, ActionKind.REMOVE, box, d.mention.entity_type.value, d.mention.mention_id))
        self.planned = {0: tuple(ops)}
        return self.planned

    def replan_hits(self, hits: Sequence[ResidualHit]) -> list[ResidualHit]:
        self.calls.append("replan_hits")
        self.hits.append(list(hits))
        return list(hits) if self.replans else []

    def report(self) -> InjectReport:
        self.calls.append("report")
        return self.reported


@dataclass
class CountingPool(InProcessPool):
    """``InProcessPool`` that counts the primary page analyses (OCR's place: the pool reads the text layer)."""

    primaries: int = 0

    def _analyse(self, ref: Any, page: int, profile: str = "primary") -> PageAnalysis:
        self.primaries += profile == "primary"
        return super()._analyse(ref, page, profile)


class Job(NamedTuple):
    result: JobResult
    inputs: FakeInputs | None
    pool: CountingPool


def _run(inputs: FakeInputs | None, residuals: tuple[BBox, ...] = ()) -> Job:
    pool = CountingPool(residuals=residuals)
    with pytest.MonkeyPatch.context() as mp:
        pool.install(mp)
        with Redactor(CONFIG) as redactor:
            result = redactor.run(LETTER, trace=True, inject=inputs)
    return Job(result, inputs, pool)


JOBS = {
    # The verification re-read finds a barcode outside every planned box, which the injection does not re-plan.
    "injected": lambda: _run(FakeInputs(PERSON), (RESIDUAL,)),
    "injected, re-planned": lambda: _run(FakeInputs(PERSON, replans=True), (RESIDUAL,)),
    "injected text only": lambda: _run(FakeInputs(PHONE, stages=False, plans=False)),
    "no injection": lambda: _run(None),
}


class Jobs:
    """Each job run once, on first use."""

    def __init__(self) -> None:
        self._done: dict[str, Job] = {}

    def __getitem__(self, name: str) -> Job:
        if name not in self._done:
            self._done[name] = JOBS[name]()
        return self._done[name]


@pytest.fixture(scope="module")
def jobs() -> Jobs:
    return Jobs()


def _inputs(job: Job) -> FakeInputs:
    assert job.inputs is not None
    return job.inputs


def _v4_metrics(result: JobResult) -> dict[str, Any]:
    (v4,) = [c for c in result.manifest["verification"]["checks"] if c["check"] == "V4_residual"]
    metrics: dict[str, Any] = v4["metrics"]
    return metrics


# ------------------------------------------------------------------------------------------- seams
def test_page_analysis_reads_the_injected_text(jobs: Jobs) -> None:
    job = jobs["injected"]
    trace = job.result.trace
    assert trace is not None and job.pool.primaries == 0 and job.pool.rereads == 1  # only verification re-reads
    tokens = list(trace.pages[0].tokens())
    assert [(t.text, t.engine) for t in tokens] == [(t.text, INJECTED_ENGINE) for t in PERSON.tokens]
    assert trace.pages[0].lines == injected_lines(PERSON, trace.pages[0].geometry)
    assert _inputs(job).calls == ["pages", "stage", "plan", "replan_hits", "report"]


def test_injected_detection_and_policy_are_the_job_s(jobs: Jobs) -> None:
    job = jobs["injected"]
    trace, staged = job.result.trace, _inputs(job).staged
    assert trace is not None and staged is not None
    assert (trace.pages, trace.decisions, trace.persons) == (staged.pages, staged.decisions, staged.detection.persons)
    assert set(trace.surrogates) == {staged.decisions[0].mention.mention_id}
    assert trace.regions == (RegionDecision(SIGNATURE, ActionKind.SYNTHESISE),)
    detectors = job.result.manifest["provenance"]["detectors"]
    assert detectors == [{"id": "bench.inject", "version": "1"}, {"id": "resolve.persons", "version": "1"}]


def test_the_injected_plan_is_what_is_drawn(jobs: Jobs) -> None:
    job = jobs["injected"]
    trace, inputs = job.result.trace, _inputs(job)
    assert trace is not None and inputs.staged is not None and inputs.planned is not None
    assert dict(trace.ops) == inputs.planned and [b.ops for b in job.pool.builds] == [inputs.planned]
    args = inputs.plan_args
    assert (args["decisions"], args["pages"], args["config"]) == (inputs.staged.decisions, trace.pages, CONFIG)
    assert args["surrogates"] == trace.surrogates
    assert args["tokens"] == {t.token_id: t for p in trace.pages for t in p.tokens()}
    assert args["region_actions"] == {SIGNATURE.region_id: ActionKind.SYNTHESISE} and args["rotated"] == frozenset()


def test_verification_re_plans_only_the_hits_the_injection_allows(jobs: Jobs) -> None:
    hit = ResidualHit(0, RESIDUAL, RegionKind.BARCODE.value, "region")
    held, replanned = jobs["injected"], jobs["injected, re-planned"]
    # Not re-planned: one output, and the hit still fails V4.
    assert _inputs(held).hits == [[hit]] and len(held.pool.builds) == 1
    assert (held.result.status, held.result.manifest["reasons"]) == (JobStatus.QUARANTINED, ["V4_RESIDUAL"])
    assert (_v4_metrics(held.result)["from_regions"], _v4_metrics(held.result)["replans"]) == (1, 0)
    # Re-planned: a second output with the residual's blackout, which passes.
    assert _inputs(replanned).hits == [[hit], []] and len(replanned.pool.builds) == 2
    trace = replanned.result.trace
    assert trace is not None and any(op.ref_id.startswith("R-") for op in trace.ops[0])
    assert (replanned.result.status, _v4_metrics(replanned.result)["replans"]) == (JobStatus.RELEASED, 1)


def test_the_trace_carries_the_injection_s_report(jobs: Jobs) -> None:
    for name in ("injected", "injected, re-planned", "injected text only"):
        job = jobs[name]
        assert job.result.trace is not None and job.result.trace.injected is _inputs(job).reported, name


def test_hooks_that_decline_leave_tool_2_s_own_stages(jobs: Jobs) -> None:
    job = jobs["injected text only"]
    trace = job.result.trace
    assert trace is not None and _inputs(job).calls == ["pages", "stage", "plan", "replan_hits", "report"]
    # Tool 2's detection read the injected text, and tool 2's planning padded the phone number it found.
    (phone,) = trace.decisions
    assert (phone.mention.entity_type, phone.mention.value, phone.action) == (
        EntityType.PHONE,
        "0412 345 678",
        ActionKind.SURROGATE,
    )
    (op,) = [op for ops in trace.ops.values() for op in ops]
    assert (op.kind, op.ref_id) == (ActionKind.SURROGATE, phone.mention.mention_id)
    assert op.ink_box is not None and op.box != op.ink_box and op.box.intersection(op.ink_box) == op.ink_box
    assert {d["id"] for d in job.result.manifest["provenance"]["detectors"]} > {"resolve.persons"}
    assert "bench.inject" not in {d["id"] for d in job.result.manifest["provenance"]["detectors"]}


def test_without_an_injection_nothing_changes(jobs: Jobs) -> None:
    job = jobs["no injection"]
    trace = job.result.trace
    assert trace is not None and trace.injected is None
    assert job.pool.primaries == 1 and INJECTED_ENGINE not in {t.engine for p in trace.pages for t in p.tokens()}
    assert job.result.status is JobStatus.RELEASED


def test_an_injection_needs_a_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = CountingPool()
    pool.install(monkeypatch)
    inputs = FakeInputs(PERSON)
    with Redactor(CONFIG) as redactor, pytest.raises(ValueError, match="trace=True"):
        redactor.run(LETTER, inject=inputs)
    assert inputs.calls == [] and pool.primaries == 0 and not pool.builds


def _imports(tree: ast.Module, *, type_checking: bool) -> Iterator[str]:
    """The modules a module imports when it is loaded (and, with ``type_checking``, for type checking only): its
    top-level imports, conditional ones included, but not those inside functions."""
    stack: list[ast.stmt] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            stack.extend([*node.body, *node.orelse] if type_checking else node.orelse)
        elif isinstance(node, ast.If | ast.With | ast.ClassDef):
            stack.extend([*node.body, *getattr(node, "orelse", [])])
        elif isinstance(node, ast.Try):
            stack.extend([*node.body, *(s for h in node.handlers for s in h.body), *node.orelse, *node.finalbody])


def _injection_modules(names: Iterable[str]) -> set[str]:
    return {n for n in names if any(n == m or n.startswith(f"{m}.") for m in INJECTION_MODULES)}


def test_production_never_loads_the_injection_modules() -> None:
    """Only the benchmark adapter and the injection itself load them: the runner names the injection's types for type
    checking, and the worker task and the ``bench`` command import what they use when they run."""
    src = ROOT / "src"
    for path in sorted((src / "mlredact").rglob("*.py")):
        module = ".".join(path.relative_to(src).with_suffix("").parts)
        if _injection_modules([module]):
            continue
        tree = ast.parse(path.read_text("utf-8"))
        assert not _injection_modules(_imports(tree, type_checking=False)), module
    runner = ast.parse((src / "mlredact" / "pipeline" / "runner.py").read_text("utf-8"))
    assert "mlredact.pipeline.injection" in _injection_modules(_imports(runner, type_checking=True))


# ------------------------------------------------------------------------------------- worker task
class Spy:
    """Stands in for region analysis: records each call, returns ``found``."""

    def __init__(self, found: PageRegions) -> None:
        self.found = found
        self.calls: list[tuple[tuple[Line, ...], dict[str, Any]]] = []

    def __call__(self, rgb: Any, lines: Sequence[Line], *args: Any, **kwargs: Any) -> PageRegions:
        self.calls.append((tuple(lines), kwargs))
        return self.found


def _analyse(
    monkeypatch: pytest.MonkeyPatch, page: InjectedPage, detect_regions: bool, found: PageRegions, **regions: Any
) -> tuple[PageAnalysis, Spy]:
    """``task_analyse_injected`` on the letter's first page, in this process, with region analysis spied on."""
    config = load_config("client", (), {**NO_MODELS, "regions": {"faces": False, **regions}})
    state: dict[str, Any] = {"config": config}
    spy = Spy(found)
    monkeypatch.setattr(workers_module, "_STATE", state)
    monkeypatch.setattr(workers_module, "analyse_regions", spy)
    try:
        with SharedBlob(LETTER) as blob:
            analysis = workers_module.task_analyse_injected(blob.ref, 0, page, detect_regions)
    finally:
        cached = state.pop("doc", None)
        if cached is not None:
            cached[1].close()
    assert set(state) == {"config"}  # no OCR engine was made
    return analysis, spy


def test_injected_analysis_finds_regions_but_keeps_every_injected_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # What a re-read and the ghost test could have changed: a line read from unexplained ink, a token taken for
    # show-through.
    (read,) = injected_lines(PHONE, A4)
    ghost = frozenset({injected_lines(PERSON, A4)[0].tokens[0].token_id})
    found = PageRegions((SIGNATURE,), extra_lines=(read,), ghost_tokens=ghost, unexplained_ink=0.25)
    analysis, spy = _analyse(monkeypatch, PERSON, True, found)
    ((lines, kwargs),) = spy.calls
    assert lines == injected_lines(PERSON, analysis.geometry) == analysis.lines  # no ghost dropped, no line added
    assert kwargs == {"detected": [], "reread": None}  # nothing re-read; no barcode on the letter, faces off
    assert (analysis.regions, analysis.unexplained_ink) == ((SIGNATURE,), 0.25)
    assert analysis.embedded_char_count > 0  # the letter's own text layer is counted, not read


def test_injected_analysis_without_region_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis, spy = _analyse(monkeypatch, PERSON, False, PageRegions((SIGNATURE,), ()))
    assert spy.calls == [] and (analysis.regions, analysis.unexplained_ink) == ((), 0.0)
    assert [t.text for t in analysis.tokens()] == [t.text for t in PERSON.tokens]


def test_an_illegible_page_still_stops_the_job_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ProcessingError) as caught:
        _analyse(monkeypatch, PERSON, True, PageRegions((), (), illegible=True), illegible_action="quarantine")
    assert (caught.value.code, caught.value.params) == (ReasonCode.PAGE_ILLEGIBLE, {"page": 0})


def test_a_broken_injected_page_is_a_content_free_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ProcessingError) as caught:
        _analyse(monkeypatch, replace(PERSON, index=1), False, PageRegions((), ()))
    assert (caught.value.code, caught.value.params) == (ReasonCode.OCR_FAILED, {"page": 0})
    assert "Zelda" not in str(caught.value) + repr(caught.value)
