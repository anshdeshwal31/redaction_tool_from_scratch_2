"""Job orchestration (plan §5): intake -> analyse -> detect -> resolve -> policy -> surrogates ->
render -> build -> verify.

Release is fail-closed: the PDF is returned only when every verification check passes.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from mlredact import __version__
from mlredact.config.loader import config_hash
from mlredact.config.schema import AppConfig
from mlredact.core.canonical import sha256_hex
from mlredact.core.errors import EnvironmentErrorMl, MlredactError, ReasonCode
from mlredact.core.geometry import BBox
from mlredact.core.model import PageAnalysis, RenderOp, Token
from mlredact.core.types import ActionKind, JobStatus, VerificationCheck
from mlredact.detect.base import Detector
from mlredact.detect.engine import DetectionResult, Seed, default_detectors, detect_document
from mlredact.detect.ner import build_ner_detectors
from mlredact.ingest.images import sniff_format
from mlredact.ingest.inspect import InspectionReport
from mlredact.manifest.build import build_manifest
from mlredact.manifest.sensitive import build_sensitive, seal
from mlredact.ocr.tesseract import binary_version
from mlredact.pdfout.builder import PageSpec, build_pdf
from mlredact.pdfout.textlayer import page_text_layer, text_layer_items
from mlredact.pipeline.pool import WorkerPool
from mlredact.pipeline.requirements import required_model_ids
from mlredact.pipeline.shared import SharedBlob
from mlredact.pipeline.workers import RedactedPage, task_analyse, task_convert_image, task_inspect, task_redact
from mlredact.policy.engine import Decision, decide
from mlredact.registry.models import load_registry, models_dir, resolve
from mlredact.render.label import NOTICE_REMOVED, NOTICE_SURROGATE
from mlredact.render.plan import check_supported, pad_box, plan_ops, rotated_token_ids
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
from mlredact.verify.checks import CheckResult, check_integrity, check_structure, check_text_leak
from mlredact.verify.residual import find_residuals, residual_ops, residual_result
from mlredact.verify.surrogates import check_surrogates

_log = get_logger("mlredact.pipeline")
IMAGE_FORMATS = frozenset({"tiff", "png", "jpeg"})


@dataclass(frozen=True)
class JobResult:
    status: JobStatus
    pdf: bytes | None = field(repr=False)
    manifest: dict[str, Any]
    run_record: dict[str, Any]
    # Sealed sensitive manifest (original text), only when the caller supplied an evaluator key.
    sensitive: bytes | None = field(default=None, repr=False)


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
    ) -> JobResult:
        """Redact ``data``.  ``scope_id`` (e.g. a matter/claim identifier) makes surrogates consistent
        across all documents of that scope; by default the scope is the document itself.
        ``sensitive_to``: an evaluator's X25519 public key; the sensitive manifest (original text) is
        then returned sealed to it (never in plaintext)."""
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
                pages = self._pool.map(task_analyse, [(blob.ref, i, "primary") for i in range(n)])
                lap("analyse_ms", t0)

                t0 = time.perf_counter()
                detection = detect_document(pages, cfg, input_sha, seeds, self._detectors)
                professionals = (
                    professional_mentions(detection.persons, detection.mentions, detection.line_view)
                    if cfg.policy.keep_professionals
                    else frozenset()
                )
                decisions = decide(detection.mentions, cfg.policy, professionals)
                removed = [d.mention for d in decisions if d.action is not ActionKind.KEEP]
                tokens = {t.token_id: t for p in pages for t in p.tokens()}
                geoms = {p.geometry.index: p.geometry for p in pages}
                originals = forbidden_keys(removed, detection.persons)
                if self._key is not None:
                    chooser = Chooser(self._key.scope_key(scope_id or input_sha))
                    factory = SurrogateFactory(chooser, originals)
                    surrogates = SurrogateAssigner(factory, detection.persons, tokens).assign(decisions)
                elif cfg.redaction.style == "label":
                    surrogates = assign_labels(decisions, detection.persons)
                all_lines = [line for p in pages for line in p.lines]
                rotated = rotated_token_ids(all_lines)
                ops = plan_ops(decisions, tokens, geoms, cfg.redaction, surrogates, rotated, all_lines)
                ops = self._with_region_ops(ops, pages)
                lap("detect_ms", t0)
                notice = None
                if cfg.output.notice:
                    notice = NOTICE_SURROGATE if cfg.redaction.style == "surrogate" else NOTICE_REMOVED

                with scrub_terms([m.value for m in removed]):
                    replans = 0
                    while True:
                        t0 = time.perf_counter()
                        redacted = self._pool.map(
                            task_redact, [(blob.ref, i, ops.get(i, ()), notice) for i in range(n)]
                        )
                        pdf = build_pdf(self._page_specs(redacted, pages, ops, notice), cfg.output)
                        lap(f"render_build_ms_{replans}", t0)

                        t0 = time.perf_counter()
                        v1 = check_structure(pdf, n, allow_fonts=cfg.output.text_layer)
                        v2 = check_text_leak(pdf, [m.value for m in removed])
                        v6, v3 = check_integrity(
                            [(r.geometry.width_pt, r.geometry.height_pt) for r in redacted],
                            pdf,
                            sum(len(v) for v in ops.values()),
                            sum(r.ops_applied for r in redacted),
                            [r.fills_ok for r in redacted],
                        )
                        v5 = (
                            self._check_surrogates(decisions, surrogates, tokens, originals)
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
                                removed,
                                self._erased_ops(ops, redacted),
                                cfg,
                                [" ".join(t for t in s.tokens if t) for s in surrogates.values()],
                                self._kept_regions(decisions, tokens, geoms),
                            )
                        lap(f"verify_ms_{replans}", t0)
                        if hits and replans < cfg.verification.max_replans:
                            replans += 1
                            ops = self._merge_ops(ops, residual_ops(hits, input_sha, replans), geoms, cfg)
                            _log.warning("verify.replan", job=job_id, iteration=replans, tokens=len(hits))
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
        return JobResult(status, pdf if released else None, manifest, run_record, sensitive)

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
            layer = None
            if self.config.output.text_layer:
                # Use the boxes that were actually erased (ink-aware refinement may differ from plan).
                erased = {(a.ref_id, a.planned): a.erased for a in r.applied}
                page_ops = [
                    replace(op, box=erased.get((op.ref_id, op.box), op.box)) for op in ops.get(page.geometry.index, ())
                ]
                items = text_layer_items(list(page.tokens()), page_ops)
                if notice and r.notice_box is not None:
                    items.append((r.notice_box, notice))
                layer = page_text_layer(r.geometry, items)
            specs.append(PageSpec(r.geometry.width_pt, r.geometry.height_pt, r.image, layer))
        return specs

    def _with_region_ops(
        self, ops: Mapping[int, tuple[RenderOp, ...]], pages: Sequence[PageAnalysis]
    ) -> dict[int, tuple[RenderOp, ...]]:
        """Non-text regions (signatures, barcodes, faces, stamps, handwriting, faint ink) -> raster ops."""
        actions = self.config.policy.region_actions
        merged: dict[int, list[RenderOp]] = {p: list(v) for p, v in ops.items()}
        for page in pages:
            for region in page.regions:
                action = actions.get(region.kind, ActionKind.REMOVE)
                if action is ActionKind.KEEP:
                    continue
                kind = ActionKind.SUPPRESS_INK if action is ActionKind.SUPPRESS_INK else ActionKind.REMOVE
                merged.setdefault(region.page, []).append(
                    RenderOp(region.page, kind, region.bbox, region.kind.value, region.region_id)
                )
        return {p: tuple(sorted(v, key=lambda o: o.sort_key)) for p, v in sorted(merged.items())}

    def _kept_regions(
        self, decisions: Sequence[Decision], tokens: Mapping[str, Token], geoms: Mapping[int, Any]
    ) -> dict[int, list[BBox]]:
        """Padded boxes of mentions kept *by override* (their type is normally acted on)."""
        actions = self.config.policy.entity_actions
        out: dict[int, list[BBox]] = {}
        for d in decisions:
            if d.action is not ActionKind.KEEP or actions[d.mention.entity_type] is ActionKind.KEEP:
                continue
            for tid in d.mention.token_ids:
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
        return check_surrogates(items, originals, shifted_dates=shifted, generalised=generalised)

    @staticmethod
    def _merge_ops(
        ops: Mapping[int, tuple[RenderOp, ...]],
        extra: Mapping[int, list[RenderOp]],
        geoms: Mapping[int, Any],
        cfg: AppConfig,
    ) -> dict[int, tuple[RenderOp, ...]]:
        merged: dict[int, list[RenderOp]] = {p: list(v) for p, v in ops.items()}
        for page, page_ops in extra.items():
            for op in page_ops:
                padded = pad_box(op.box, op.box.height, cfg.redaction, geoms[page].bounds)
                merged.setdefault(page, []).append(RenderOp(op.page, op.kind, padded, op.reason, op.ref_id))
        return {p: tuple(sorted(v, key=lambda o: o.sort_key)) for p, v in sorted(merged.items())}


def _reason_for(check: CheckResult) -> str:
    return {
        "V1_structure": ReasonCode.VERIFY_STRUCTURE,
        "V2_text": ReasonCode.VERIFY_TEXT_LEAK,
        "V3_pixels": ReasonCode.VERIFY_PIXELS,
        "V4_residual": ReasonCode.VERIFY_RESIDUAL,
        "V5_surrogate": ReasonCode.VERIFY_SURROGATE,
        "V6_integrity": ReasonCode.VERIFY_INTEGRITY,
    }[check.check.value].value
