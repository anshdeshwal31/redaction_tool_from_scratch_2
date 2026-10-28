"""Job orchestration (plan §5): intake -> analyse -> detect -> resolve -> policy -> surrogates ->
render -> build -> verify.

Release is fail-closed: the PDF is returned only when every verification check passes.  On request
a job also returns its trace (what every stage saw and decided) for the benchmark adapter, and takes the
benchmark's injected stage inputs (``mlredact.pipeline.injection``).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from mlredact import __version__
from mlredact.config.loader import config_hash
from mlredact.config.schema import AppConfig
from mlredact.core.canonical import sha256_hex
from mlredact.core.errors import EnvironmentErrorMl, MlredactError, ReasonCode
from mlredact.core.geometry import BBox
from mlredact.core.model import Line, Mention, PageAnalysis, PageGeometry, Region, RenderOp, Token
from mlredact.core.types import ActionKind, EntityType, JobStatus, VerificationCheck
from mlredact.detect.base import Detector
from mlredact.detect.engine import DetectionResult, Seed, default_detectors, detect_document
from mlredact.detect.ner import build_ner_detectors
from mlredact.ingest.images import sniff_format
from mlredact.ingest.inspect import InspectionReport
from mlredact.manifest.build import build_manifest
from mlredact.manifest.sensitive import build_sensitive, seal
from mlredact.ocr.tesseract import binary_version
from mlredact.pdfout.builder import PageSpec, build_pdf, build_text_pdf
from mlredact.pdfout.textlayer import LayerMask, page_text_layer, text_layer_items
from mlredact.pipeline.pool import WorkerPool
from mlredact.pipeline.requirements import required_model_ids
from mlredact.pipeline.shared import SharedBlob
from mlredact.pipeline.workers import (
    RedactedPage,
    task_analyse,
    task_analyse_injected,
    task_convert_image,
    task_inspect,
    task_redact,
)
from mlredact.policy.addresses import (
    AddressJoin,
    JoinedUnit,
    KeptRecord,
    MentionLayout,
    fold_word,
    join_addresses,
)
from mlredact.policy.engine import Decision, decide
from mlredact.registry.models import load_registry, models_dir, resolve
from mlredact.render.label import NOTICE_REMOVED, NOTICE_SURROGATE
from mlredact.render.plan import check_supported, pad_box, plan_ops, rotated_token_ids
from mlredact.resolve.persons import PersonResolution
from mlredact.resolve.roles import professional_mentions
from mlredact.resources.loader import verify_resources
from mlredact.runtime.profile import RuntimeProfile, check_runtime_profile
from mlredact.security.exceptions import sanitize
from mlredact.security.logging import get_logger, scrub_terms
from mlredact.surrogate.assign import MentionSurrogate, SurrogateAssigner, forbidden_keys
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.generators import SurrogateFactory
from mlredact.surrogate.keys import MasterKey, load_master_key
from mlredact.surrogate.labels import assign_labels
from mlredact.verify.addresses import AddressFakes, AddressReread, reread_of
from mlredact.verify.checks import CheckResult, check_integrity, check_structure, check_text_leak
from mlredact.verify.needles import MASK, AddressNeedles, AddressText, needles_of
from mlredact.verify.residual import find_residuals, residual_ops, residual_result
from mlredact.verify.surrogates import check_surrogates

if TYPE_CHECKING:
    from mlredact.pipeline.injection import InjectReport, StageInputs

_log = get_logger("mlredact.pipeline")
IMAGE_FORMATS = frozenset({"tiff", "png", "jpeg"})
# The last output built and what it was drawn from: (pdf, render ops, rendered pages).
_Built = tuple[bytes, dict[int, tuple[RenderOp, ...]], list[RedactedPage]]


@dataclass(frozen=True, slots=True)
class RegionDecision:
    """What happens to a non-text region: KEEP, or the raster operation drawn over it."""

    region: Region
    action: ActionKind


@dataclass(frozen=True)
class JobTrace:
    """What every stage of one job saw and decided, for the benchmark adapter (``mlredact bench``) only.

    It holds document text (tokens, mention values, surrogates): it is never logged, never part of the
    manifest, and only the adapter writes it out, into the evaluator's run folder.  ``pdf``, ``ops``
    and ``erased`` belong together: the last output the pipeline built (released or not, also when a
    later step failed), the render operations it was drawn from (after verification re-plans) and the
    boxes those operations actually erased, keyed like the manifest's ``erased_px``.  ``injected``: the
    report of the job's injected stage inputs (benchmark oracle modes), None without them.  The address join's
    output (``addresses: whole``, the pipeline's own detection; empty otherwise): ``layouts`` (mention id -> where the
    mention sits in its unit), ``units`` (unit id -> unit), ``kept_places`` (the places mentioned on their own, as
    the grammar recorded them) and ``address_fakes`` (unit id -> token id -> the text drawn, '' erases a token)."""

    pdf: bytes | None = field(repr=False)
    pages: tuple[PageAnalysis, ...] = field(repr=False)
    decisions: tuple[Decision, ...] = field(repr=False)
    persons: PersonResolution | None = field(repr=False)
    surrogates: Mapping[str, MentionSurrogate] = field(repr=False)
    ops: Mapping[int, tuple[RenderOp, ...]] = field(repr=False)
    erased: Mapping[tuple[str, BBox], BBox]
    regions: tuple[RegionDecision, ...]
    injected: InjectReport | None = field(default=None, repr=False)
    layouts: Mapping[str, MentionLayout] = field(default_factory=dict, repr=False)
    units: Mapping[str, JoinedUnit] = field(default_factory=dict, repr=False)
    kept_places: tuple[KeptRecord, ...] = field(default=(), repr=False)
    address_fakes: Mapping[str, Mapping[str, str]] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class _AddressChecks:
    """What verification knows of the document's addresses, built once per job: ``removed``, the removed mentions that
    V2 and V4 search by value (a unit's mention is searched by its targets); ``scrub``, the units' targets for the log
    scrubber; ``needles`` and ``mask`` (V2); ``fakes`` (V5); ``reread`` (V4); ``kept``, the regions V4's
    re-detection leaves alone."""

    removed: list[Mention]
    scrub: list[str]
    needles: AddressNeedles
    mask: LayerMask
    fakes: AddressFakes | None
    reread: AddressReread | None
    kept: dict[int, list[BBox]]


@dataclass(frozen=True)
class JobResult:
    status: JobStatus
    pdf: bytes | None = field(repr=False)
    manifest: dict[str, Any]
    run_record: dict[str, Any]
    # Sealed sensitive manifest (original text), only when the caller supplied an evaluator key.
    sensitive: bytes | None = field(default=None, repr=False)
    # The redacted copy of a quarantined job, only when the caller asked to keep it (testing and
    # review).  It failed verification: never a release.
    held_back_pdf: bytes | None = field(default=None, repr=False)
    # Only when the caller asked for it (benchmark adapter): holds document text.
    trace: JobTrace | None = field(default=None, repr=False)


class Redactor:
    """Long-lived redaction service object: validates the environment once, keeps workers warm."""

    def __init__(self, config: AppConfig, *, workers: int | None = None, log_level: str = "WARNING") -> None:
        self.config = config
        self.config_hash = config_hash(config)
        self.runtime: RuntimeProfile = check_runtime_profile(config)
        verify_resources()
        check_supported(config.redaction.style, list(config.policy.entity_actions.values()))
        self._key: MasterKey | None = None
        if config.redaction.style == "surrogate":
            self._key = load_master_key(
                key_file=config.surrogate.key_file,
                allow_development_key=config.surrogate.allow_development_key,
                production=config.runtime.expected_profile is not None,
            )
        directory = models_dir(config.runtime.models_dir)
        registry = load_registry()
        self._model_ids = required_model_ids(config)
        for mid in self._model_ids:
            resolve(mid, directory)  # fail before spawning workers if anything is missing/tampered
        if config.ocr.engine_b.enabled and binary_version(config.ocr.engine_b.binary) is None:
            raise EnvironmentErrorMl(ReasonCode.MODEL_MISSING)  # engine B's binary is not installed
        self._models = [registry[mid].provenance() for mid in self._model_ids]
        # Document-level detectors (rules + NER) live in this process; NER models load once here.
        self._detectors: tuple[Detector, ...] = (*default_detectors(), *build_ner_detectors(config, directory))
        self._pool = WorkerPool(config, workers, log_level=log_level)

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> Redactor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --------------------------------------------------------------------------------------- run
    def run(
        self,
        data: bytes,
        *,
        seeds: Sequence[Seed] = (),
        scope_id: str | None = None,
        sensitive_to: bytes | None = None,
        keep_held_back: bool = False,
        trace: bool = False,
        inject: StageInputs | None = None,
    ) -> JobResult:
        """Redact ``data``.  ``scope_id`` (e.g. a matter/claim identifier) makes surrogates consistent
        across all documents of that scope; by default the scope is the document itself.
        ``sensitive_to``: an evaluator's X25519 public key; the sensitive manifest (original text) is
        then returned sealed to it (never in plaintext).
        ``keep_held_back``: when the job is quarantined after its output was built, return that output
        as ``held_back_pdf`` (testing and review only; ``pdf`` stays None).
        ``trace``: also return the job's :class:`JobTrace` (document text included) for the benchmark
        adapter.  Everything else in the result is the same with or without it.
        ``inject``: the benchmark's injected stage inputs (``mlredact bench`` oracle modes only; they need
        ``trace``).  Without them every stage is the pipeline's own."""
        if inject is not None and not trace:
            raise ValueError("injected stage inputs are for the benchmark adapter: they need trace=True")
        cfg = self.config
        job_id = f"job-{uuid.uuid4().hex}"
        input_sha = sha256_hex(data)
        timings: dict[str, float] = {}
        t_job = time.perf_counter()
        inspection: InspectionReport | None = None
        pages: list[PageAnalysis] = []
        detection: DetectionResult | None = None
        decisions: list[Decision] = []
        surrogates: dict[str, MentionSurrogate] = {}
        ops: dict[int, tuple[RenderOp, ...]] = {}
        checks: list[CheckResult] = []
        reasons: list[str] = []
        pdf: bytes | None = None
        redacted: list[RedactedPage] = []
        built: _Built | None = None  # read only for the trace
        join = AddressJoin(())  # the address join (empty while the mentions are injected)
        address_fakes: Mapping[str, Mapping[str, str]] = {}
        _log.info("job.start", job=job_id, doc=input_sha[:32])

        def lap(name: str, t0: float) -> None:
            timings[name] = round((time.perf_counter() - t0) * 1000.0, 1)

        source_format = sniff_format(data) or "unknown"
        try:
            pdf_input = data
            if source_format in IMAGE_FORMATS:
                t0 = time.perf_counter()
                with SharedBlob(data) as raw:
                    pdf_input = self._pool.call(task_convert_image, raw.ref)
                lap("convert_ms", t0)
            with SharedBlob(pdf_input) as blob:
                t0 = time.perf_counter()
                inspection = self._pool.call(task_inspect, blob.ref)
                lap("inspect_ms", t0)
                n = inspection.page_count

                t0 = time.perf_counter()
                if inject is None:
                    pages = self._pool.map(task_analyse, [(blob.ref, i, "primary") for i in range(n)])
                else:
                    pages = self._pool.map(
                        task_analyse_injected,
                        [(blob.ref, i, page, inject.detect_regions) for i, page in enumerate(inject.pages(n))],
                    )
                lap("analyse_ms", t0)

                t0 = time.perf_counter()
                staged = inject.stage(pages, cfg, input_sha) if inject is not None else None
                if staged is None:
                    detection = detect_document(pages, cfg, input_sha, seeds, self._detectors)
                    professionals = (
                        professional_mentions(detection.persons, detection.mentions, detection.line_view)
                        if cfg.policy.keep_professionals
                        else frozenset()
                    )
                    tokens = {t.token_id: t for p in pages for t in p.tokens()}
                    # addresses: whole -> the join (units, kept places); parts -> the per-part split of today.
                    join = join_addresses(detection.mentions, pages, cfg.policy)
                    decisions = decide(join.mentions, cfg.policy, professionals)
                else:
                    pages, detection, decisions = list(staged.pages), staged.detection, list(staged.decisions)
                    tokens = {t.token_id: t for p in pages for t in p.tokens()}
                removed = [d.mention for d in decisions if d.action is not ActionKind.KEEP]
                geoms = {p.geometry.index: p.geometry for p in pages}
                originals = forbidden_keys(removed, detection.persons)
                if self._key is not None:
                    chooser = Chooser(self._key.scope_key(scope_id or input_sha))
                    factory = SurrogateFactory(chooser, originals)
                    assigner = SurrogateAssigner(
                        factory,
                        detection.persons,
                        tokens,
                        layouts=join.layouts,
                        units=join.units,
                        address_state=cfg.policy.address_state,
                    )
                    surrogates = assigner.assign(decisions)
                    address_fakes = assigner.address_fakes
                elif cfg.redaction.style == "label":
                    surrogates = assign_labels(decisions, detection.persons)
                all_lines = [line for p in pages for line in p.lines]
                rotated = rotated_token_ids(all_lines)
                planned = (
                    inject.plan(
                        decisions=decisions,
                        surrogates=surrogates,
                        tokens=tokens,
                        pages=pages,
                        region_actions={d.region.region_id: d.action for p in pages for d in self._region_decisions(p)},
                        rotated=rotated,
                        config=cfg,
                    )
                    if inject is not None
                    else None
                )
                if planned is None:
                    ops = plan_ops(decisions, tokens, geoms, cfg.redaction, surrogates, rotated, all_lines)
                    ops = self._with_region_ops(ops, pages)
                else:
                    ops = planned
                lap("detect_ms", t0)
                notice = None
                if cfg.output.notice:
                    notice = NOTICE_SURROGATE if cfg.redaction.style == "surrogate" else NOTICE_REMOVED

                checks_of = self._address_checks(
                    join, decisions, removed, surrogates, tokens, address_fakes, geoms, all_lines
                )
                with scrub_terms([*(m.value for m in checks_of.removed), *checks_of.scrub]):
                    replans = 0
                    while True:
                        t0 = time.perf_counter()
                        redacted = self._pool.map(
                            task_redact, [(blob.ref, i, ops.get(i, ()), notice) for i in range(n)]
                        )
                        pdf = build_pdf(self._page_specs(redacted, pages, ops, notice), cfg.output)
                        built = (pdf, ops, redacted)
                        lap(f"render_build_ms_{replans}", t0)

                        t0 = time.perf_counter()
                        v1 = check_structure(pdf, n, allow_fonts=cfg.output.text_layer)
                        v2 = check_text_leak(
                            pdf,
                            [m.value for m in checks_of.removed],
                            addresses=self._address_text(checks_of, redacted, pages, ops, notice),
                        )
                        v6, v3 = check_integrity(
                            [(r.geometry.width_pt, r.geometry.height_pt) for r in redacted],
                            pdf,
                            sum(len(v) for v in ops.values()),
                            sum(r.ops_applied for r in redacted),
                            [r.fills_ok for r in redacted],
                        )
                        v5 = (
                            self._check_surrogates(decisions, surrogates, tokens, originals, checks_of.fakes)
                            if cfg.redaction.style == "surrogate"
                            else CheckResult(VerificationCheck.SURROGATE, True, {"surrogates": 0})
                        )
                        hits = []
                        if cfg.verification.re_ocr:
                            with SharedBlob(pdf) as out_blob:
                                out_pages = self._pool.map(
                                    task_analyse, [(out_blob.ref, i, "verify") for i in range(n)]
                                )
                            hits = find_residuals(
                                out_pages,
                                checks_of.removed,
                                self._erased_ops(ops, redacted),
                                cfg,
                                [" ".join(t for t in s.tokens if t) for s in surrogates.values()],
                                checks_of.kept,
                                addresses=checks_of.reread,
                            )
                        lap(f"verify_ms_{replans}", t0)
                        # Injected stage inputs may re-plan fewer hits; every hit still counts for the verdict.
                        replan = hits if inject is None else inject.replan_hits(hits)
                        if replan and replans < cfg.verification.max_replans:
                            replans += 1
                            ops = self._merge_ops(
                                ops, residual_ops(replan, input_sha, replans), geoms, cfg, checks_of.mask
                            )
                            _log.warning("verify.replan", job=job_id, iteration=replans, tokens=len(replan))
                            continue
                        checks = [v1, v2, v3, residual_result(hits, replans), v5, v6]
                        break
        except MlredactError as exc:
            info = sanitize(exc)
            reasons.append(info.code.value)
            _log.error("job.failed", job=job_id, **info.as_fields())
            pdf = None
        except Exception as exc:  # never let an unexpected error release output
            info = sanitize(exc)
            reasons.append(ReasonCode.INTERNAL.value)
            _log.error("job.failed", job=job_id, **info.as_fields())
            pdf = None

        failed = [c for c in checks if not c.passed]
        reasons.extend(_reason_for(c) for c in failed)
        released = pdf is not None and bool(checks) and not failed
        status = JobStatus.RELEASED if released else JobStatus.QUARANTINED
        manifest = build_manifest(
            status=status,
            input_sha256=input_sha,
            config=cfg,
            config_hash=self.config_hash,
            runtime=self.runtime,
            models=self._models,
            detectors=detection.detectors if detection else (),
            inspection=inspection,
            pages=pages,
            decisions=decisions,
            ops=ops,
            low_evidence=detection.low_evidence if detection else (),
            checks=checks,
            reasons=reasons,
            propagation_rounds=detection.propagation_rounds if detection else 0,
            surrogates=surrogates,
            replacement_field="label" if cfg.redaction.style == "label" else "surrogate",
            surrogate_key_id=self._key.key_id if self._key else None,
            surrogate_scope="caller" if scope_id else "document",
            erased={(a.ref_id, a.planned): a.erased for r in redacted for a in r.applied},
            notices={r.geometry.index: r.notice_box is not None for r in redacted},
            source_format=source_format,
        )
        timings["total_ms"] = round((time.perf_counter() - t_job) * 1000.0, 1)
        run_record = {
            "job_id": job_id,
            "mlredact_version": __version__,
            "status": status.value,
            "reasons": sorted(reasons),
            "workers": self._pool.workers,
            "timings_ms": timings,
        }
        sensitive = None
        if sensitive_to is not None and detection is not None:
            sensitive = seal(
                build_sensitive(
                    input_sha256=input_sha,
                    pages=pages,
                    decisions=decisions,
                    surrogates=surrogates,
                    persons=detection.persons,
                    low_evidence=detection.low_evidence,
                ),
                sensitive_to,
            )
        _log.info("job.done", job=job_id, status=status, pages=len(pages), duration_ms=timings["total_ms"])
        held_back = pdf if keep_held_back and not released else None
        job_trace = (
            self._trace(
                pages,
                decisions,
                detection,
                surrogates,
                built,
                ops,
                redacted,
                injected=inject.report() if inject is not None else None,
                join=join,
                address_fakes=address_fakes,
            )
            if trace
            else None
        )
        return JobResult(status, pdf if released else None, manifest, run_record, sensitive, held_back, job_trace)

    # ------------------------------------------------------------------------------- helpers
    def _page_specs(
        self,
        redacted: Sequence[RedactedPage],
        pages: Sequence[PageAnalysis],
        ops: Mapping[int, Sequence[RenderOp]],
        notice: str | None,
    ) -> list[PageSpec]:
        specs = []
        for r, page in zip(redacted, pages, strict=True):
            layer = self._text_layer(r, page, ops, notice, None) if self.config.output.text_layer else None
            specs.append(PageSpec(r.geometry.width_pt, r.geometry.height_pt, r.image, layer))
        return specs

    @staticmethod
    def _text_layer(
        r: RedactedPage,
        page: PageAnalysis,
        ops: Mapping[int, Sequence[RenderOp]],
        notice: str | None,
        mask: LayerMask | None,
    ) -> bytes | None:
        """One page's invisible text layer; with ``mask``, its exempt rendition (verification V2)."""
        # Use the boxes that were actually erased (ink-aware refinement may differ from plan).
        erased = {(a.ref_id, a.planned): a.erased for a in r.applied}
        page_ops = [replace(op, box=erased.get((op.ref_id, op.box), op.box)) for op in ops.get(page.geometry.index, ())]
        items = text_layer_items(list(page.tokens()), page_ops, mask)
        if notice and r.notice_box is not None:
            items.append((r.notice_box, notice))
        return page_text_layer(r.geometry, items)

    def _address_text(
        self,
        checks: _AddressChecks,
        redacted: Sequence[RedactedPage],
        pages: Sequence[PageAnalysis],
        ops: Mapping[int, Sequence[RenderOp]],
        notice: str | None,
    ) -> AddressText | None:
        """The address needles of V2 with the exempt rendition of the text layer just built (None: no units)."""
        if not checks.needles:
            return None
        if not self.config.output.text_layer:
            return AddressText(checks.needles)
        layers = [
            (r.geometry.width_pt, r.geometry.height_pt, self._text_layer(r, page, ops, notice, checks.mask))
            for r, page in zip(redacted, pages, strict=True)
        ]
        return AddressText(checks.needles, build_text_pdf(layers))

    def _address_checks(
        self,
        join: AddressJoin,
        decisions: Sequence[Decision],
        removed: Sequence[Mention],
        surrogates: Mapping[str, MentionSurrogate],
        tokens: Mapping[str, Token],
        fakes: Mapping[str, Mapping[str, str]],
        geoms: Mapping[int, PageGeometry],
        lines: Sequence[Line],
    ) -> _AddressChecks:
        """What verification knows of the document's addresses (TOOL2 section 6): the removed mentions it still searches
        by value (a unit's mention is searched by its targets instead), the exempt positions, the V2 mask, V5's fakes
        and V4's data."""
        plain = [m for m in removed if m.mention_id not in join.layouts]
        kept = self._kept_regions(decisions, tokens, geoms, join.exempt.kept_places)
        exempt_ids = frozenset(join.exempt.kept_places) | {
            t for d in decisions if d.action is ActionKind.KEEP for t in d.mention.token_ids
        }
        replacement = {
            tid: text
            for d in decisions
            if d.action is not ActionKind.KEEP and (sur := surrogates.get(d.mention.mention_id)) is not None
            for tid, text in zip(d.mention.token_ids, sur.tokens, strict=True)
        }
        ink: dict[tuple[int, BBox], Sequence[Token]] = {}
        for line in lines:
            if line.tokens:
                box = line.tokens[0].bbox
                for t in line.tokens[1:]:
                    box = box.union(t.bbox)
                ink[(line.page, box)] = line.tokens
        mask = LayerMask(exempt_ids, replacement, ink, MASK)
        if not join.units:
            return _AddressChecks(plain, [], AddressNeedles(), mask, None, None, kept)
        targets = [join.targets[u] for u in sorted(join.targets)]
        applied = frozenset(
            t
            for d in decisions
            if d.mention.mention_id in join.layouts and d.mention.mention_id in surrogates
            for t in d.mention.token_ids
        )
        texts = {t: tokens[t].text for u in join.units.values() for t in u.roles}
        fake_words = frozenset(
            w for sur in surrogates.values() for tok in sur.tokens for w in map(fold_word, tok.split()) if w
        )
        reread = (
            reread_of(
                join.units,
                join.targets,
                texts,
                fakes,
                applied,
                fake_words,
                self._boxes(exempt_ids, tokens, geoms),
                (tok.text for tid, tok in tokens.items() if tid not in texts),
            )
            if self.config.policy.addresses == "whole" and self.config.verification.re_ocr
            else None
        )
        terms = [" ".join(words) for t in targets for words in t.compound] + [w for t in targets for w in t.whole]
        return _AddressChecks(
            plain,
            sorted({term for term in terms if len(term) >= 3}),
            needles_of(targets),
            mask,
            AddressFakes(join.units, texts, fakes, applied) if fakes else None,
            reread,
            kept,
        )

    def _region_decisions(self, page: PageAnalysis) -> list[RegionDecision]:
        """Non-text regions (signatures, barcodes, faces, stamps, handwriting, faint ink) -> actions.

        Synthetic stand-ins belong with realistic surrogates; the label and blackout styles remove."""
        actions = self.config.policy.region_actions
        out: list[RegionDecision] = []
        for region in page.regions:
            action = actions.get(region.kind, ActionKind.REMOVE)
            if action is not ActionKind.KEEP:
                drawn = action is ActionKind.SUPPRESS_INK or (
                    action is ActionKind.SYNTHESISE and self.config.redaction.style == "surrogate"
                )
                action = action if drawn else ActionKind.REMOVE
            out.append(RegionDecision(region, action))
        return out

    def _with_region_ops(
        self, ops: Mapping[int, tuple[RenderOp, ...]], pages: Sequence[PageAnalysis]
    ) -> dict[int, tuple[RenderOp, ...]]:
        """Region decisions -> raster ops."""
        merged: dict[int, list[RenderOp]] = {p: list(v) for p, v in ops.items()}
        for page in pages:
            for d in self._region_decisions(page):
                if d.action is ActionKind.KEEP:
                    continue
                r = d.region
                merged.setdefault(r.page, []).append(RenderOp(r.page, d.action, r.bbox, r.kind.value, r.region_id))
        return {p: tuple(sorted(v, key=lambda o: o.sort_key)) for p, v in sorted(merged.items())}

    def _trace(
        self,
        pages: Sequence[PageAnalysis],
        decisions: Sequence[Decision],
        detection: DetectionResult | None,
        surrogates: Mapping[str, MentionSurrogate],
        built: _Built | None,
        ops: Mapping[int, tuple[RenderOp, ...]],
        redacted: Sequence[RedactedPage],
        injected: InjectReport | None = None,
        join: AddressJoin | None = None,
        address_fakes: Mapping[str, Mapping[str, str]] | None = None,
    ) -> JobTrace:
        """The job's trace.  When an output was built, the ops and erased boxes are the ones it was
        drawn from (a step that failed after it may already have planned more)."""
        pdf: bytes | None = None
        if built is not None:
            pdf, ops, redacted = built
        return JobTrace(
            pdf=pdf,
            pages=tuple(pages),
            decisions=tuple(decisions),
            persons=detection.persons if detection else None,
            surrogates=dict(surrogates),
            ops=dict(ops),
            erased={(a.ref_id, a.planned): a.erased for r in redacted for a in r.applied},
            regions=tuple(d for p in pages for d in self._region_decisions(p)),
            injected=injected,
            layouts=dict(join.layouts) if join is not None else {},
            units=dict(join.units) if join is not None else {},
            kept_places=join.kept if join is not None else (),
            address_fakes=dict(address_fakes) if address_fakes else {},
        )

    def _kept_regions(
        self,
        decisions: Sequence[Decision],
        tokens: Mapping[str, Token],
        geoms: Mapping[int, Any],
        places: Collection[str] = (),
    ) -> dict[int, list[BBox]]:
        """Padded boxes of mentions kept *by override* (their type is normally acted on).  Under ``addresses: whole``
        also the kept LOCALITY and POSTCODE mentions and the places mentioned on their own (``places``: the join's
        record, less what a mention replaces), which verification leaves alone."""
        actions = self.config.policy.entity_actions
        whole = self.config.policy.addresses == "whole"
        ids: set[str] = set(places) if whole else set()
        for d in decisions:
            if d.action is not ActionKind.KEEP:
                continue
            etype = d.mention.entity_type
            by_type = actions[etype] is ActionKind.KEEP
            if by_type and not (whole and etype in (EntityType.LOCALITY, EntityType.POSTCODE)):
                continue
            ids.update(d.mention.token_ids)
        return self._boxes(ids, tokens, geoms)

    def _boxes(
        self, token_ids: Collection[str], tokens: Mapping[str, Token], geoms: Mapping[int, Any]
    ) -> dict[int, list[BBox]]:
        """The padded boxes of tokens, by page, in a fixed order."""
        out: dict[int, list[BBox]] = {}
        for tid in sorted(token_ids):
            tok = tokens[tid]
            box = pad_box(tok.bbox, tok.bbox.height, self.config.redaction, geoms[tok.page].bounds)
            out.setdefault(tok.page, []).append(box)
        return out

    @staticmethod
    def _erased_ops(
        ops: Mapping[int, Sequence[RenderOp]], redacted: Sequence[RedactedPage]
    ) -> dict[int, list[RenderOp]]:
        """The planned ops with their boxes replaced by the boxes actually erased by the workers."""
        erased = {(a.ref_id, a.planned): a.erased for r in redacted for a in r.applied}
        return {
            page: [replace(op, box=erased.get((op.ref_id, op.box), op.box)) for op in page_ops]
            for page, page_ops in ops.items()
        }

    @staticmethod
    def _check_surrogates(
        decisions: Sequence[Decision],
        surrogates: Mapping[str, MentionSurrogate],
        tokens: Mapping[str, Token],
        originals: set[str],
        addresses: AddressFakes | None = None,
    ) -> CheckResult:
        if not surrogates:
            return CheckResult(VerificationCheck.SURROGATE, True, {"surrogates": 0})
        items = []
        shifted: list[tuple[str, str]] = []
        generalised: list[str] = []
        for d in decisions:
            sur = surrogates.get(d.mention.mention_id)
            if sur is None:
                continue
            text = " ".join(s for s in sur.tokens if s)
            if d.action is ActionKind.DATE_SHIFT:
                shifted.append((d.mention.value, text))
                continue
            if d.action is ActionKind.GENERALISE:
                generalised.append(text)
                continue
            orig = [tokens[t].text for t in d.mention.token_ids]
            replaced = tuple(s for s, o in zip(sur.tokens, orig, strict=True) if s and s != o)
            items.append((d.mention.entity_type, text, replaced))
        return check_surrogates(items, originals, shifted_dates=shifted, generalised=generalised, addresses=addresses)

    @staticmethod
    def _merge_ops(
        ops: Mapping[int, tuple[RenderOp, ...]],
        extra: Mapping[int, list[RenderOp]],
        geoms: Mapping[int, Any],
        cfg: AppConfig,
        mask: LayerMask | None = None,
    ) -> dict[int, tuple[RenderOp, ...]]:
        """The ops with a re-plan's operations added.  A re-typeset line a re-plan draws over takes the re-plan's words
        in place of the ones it replaces (``mask`` holds the lines and what was drawn in place of their tokens): the
        text layer must not keep the words the page no longer shows."""
        merged: dict[int, list[RenderOp]] = {p: list(v) for p, v in ops.items()}
        for page, page_ops in extra.items():
            for op in page_ops:
                padded = pad_box(op.box, op.box.height, cfg.redaction, geoms[page].bounds)
                merged.setdefault(page, []).append(replace(op, box=padded))
        if mask is not None:
            for page, page_ops in extra.items():
                merged[page] = [
                    _retext(o, page_ops, mask) if o.kind is ActionKind.RETYPESET_REGION else o for o in merged[page]
                ]
        return {p: tuple(sorted(v, key=lambda o: o.sort_key)) for p, v in sorted(merged.items())}


def _retext(op: RenderOp, replans: Sequence[RenderOp], mask: LayerMask) -> RenderOp:
    """A re-typeset line with the words under a re-plan's operations replaced by the operation's text (or dropped)."""
    line = mask.lines.get((op.page, op.ink_box)) if op.ink_box is not None else None
    if line is None:
        return op
    words: list[str] = []
    used: set[int] = set()
    changed = False
    for tok in line:
        under = next(
            (
                i
                for i, r in enumerate(replans)
                if (inter := tok.bbox.intersection(r.box)) is not None and inter.area * 2 >= tok.bbox.area
            ),
            None,
        )
        if under is None:
            words.append(mask.replacement.get(tok.token_id, tok.text))
            continue
        changed = True
        if under not in used:
            used.add(under)
            words.append(replans[under].text or "")
    return replace(op, text=" ".join(w for w in words if w)) if changed else op


def _reason_for(check: CheckResult) -> str:
    return {
        "V1_structure": ReasonCode.VERIFY_STRUCTURE,
        "V2_text": ReasonCode.VERIFY_TEXT_LEAK,
        "V3_pixels": ReasonCode.VERIFY_PIXELS,
        "V4_residual": ReasonCode.VERIFY_RESIDUAL,
        "V5_surrogate": ReasonCode.VERIFY_SURROGATE,
        "V6_integrity": ReasonCode.VERIFY_INTEGRITY,
    }[check.check.value].value
