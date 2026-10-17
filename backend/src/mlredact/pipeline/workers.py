"""Worker-process tasks.  All untrusted PDF parsing and all numeric inference happen here.

Each task is a pure function of (document bytes, page index, config, pinned models).  Exceptions
leaving a worker are re-raised as content-free :class:`MlredactError` subclasses, so nothing
from a third-party exception message can travel back to the parent.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mlredact.config.schema import AppConfig
from mlredact.core.errors import MlredactError, ProcessingError, ReasonCode
from mlredact.core.geometry import BBox, union_all
from mlredact.core.model import Line, PageAnalysis, PageGeometry, Region, RenderOp
from mlredact.core.types import PageKind
from mlredact.ingest.inspect import InspectionReport, inspect_pdf
from mlredact.ingest.render import open_document, render_page
from mlredact.layout.barcodes import find_barcodes
from mlredact.layout.regions import analyse_regions
from mlredact.ocr.fusion import fuse_embedded, fuse_tesseract
from mlredact.pdfout.encode import EncodedImage, encode_raster
from mlredact.pipeline.shared import BlobRef, read_blob
from mlredact.registry.models import models_dir, resolve
from mlredact.render.raster import Applied, apply_ops
from mlredact.runtime.determinism import apply_process_settings
from mlredact.security.logging import configure_logging

_STATE: dict[str, Any] = {}


def init_worker(config_json: str, log_level: str, strict_logging: bool) -> None:
    config = AppConfig.model_validate_json(config_json)
    apply_process_settings(config.runtime.threads)
    configure_logging(log_level, strict=strict_logging)
    with contextlib.suppress(AttributeError, OSError):
        os.nice(5)  # type: ignore[attr-defined]  # be a good neighbour on shared hosts (POSIX only)
    _STATE.clear()
    _STATE["config"] = config


def _config() -> AppConfig:
    cfg = _STATE.get("config")
    if cfg is None:  # pragma: no cover - initializer always runs first
        raise ProcessingError(ReasonCode.INTERNAL)
    return cfg  # type: ignore[no-any-return]


def _engine(profile: str) -> Any:
    key = f"ocr:{profile}"
    if key not in _STATE:
        from mlredact.ocr.ppocr import PPOCREngine

        cfg = _config()
        engine_cfg = cfg.ocr.engine_a if profile == "primary" else cfg.verification.ocr
        _STATE[key] = PPOCREngine(engine_cfg, cfg.runtime.threads, models_dir(cfg.runtime.models_dir))
    return _STATE[key]


def _document(ref: BlobRef) -> Any:
    cached = _STATE.get("doc")
    if cached is not None and cached[0] == ref.key:
        return cached[1]
    if cached is not None:
        with contextlib.suppress(Exception):
            cached[1].close()
    doc = open_document(read_blob(ref), _config().render)
    _STATE["doc"] = (ref.key, doc)
    return doc


def _guard(code: ReasonCode, page: int | None = None) -> ProcessingError:
    return ProcessingError(code) if page is None else ProcessingError(code, page=page)


def task_engine_version(profile: str) -> str:
    return str(_engine(profile).version)


def task_convert_image(ref: BlobRef) -> bytes:
    """TIFF / PNG / JPEG input -> image-only PDF (decoding untrusted images happens here, not in
    the parent process)."""
    from mlredact.ingest.images import image_to_pdf

    try:
        return image_to_pdf(read_blob(ref), _config().limits)
    except MlredactError:
        raise
    except Exception:
        raise _guard(ReasonCode.INPUT_UNREADABLE) from None


def task_inspect(ref: BlobRef) -> InspectionReport:
    try:
        return inspect_pdf(read_blob(ref), _config().limits)
    except MlredactError:
        raise
    except Exception:
        raise _guard(ReasonCode.INPUT_UNREADABLE) from None


def _page_kind(embedded_chars: int, images: int) -> PageKind:
    if embedded_chars == 0 and images == 0:
        return PageKind.EMPTY
    if embedded_chars == 0:
        return PageKind.IMAGE_ONLY
    return PageKind.MIXED if images else PageKind.BORN_DIGITAL


def _tesseract() -> Any:
    if "tesseract" not in _STATE:
        from mlredact.ocr.tesseract import TesseractEngine

        cfg = _config()
        _STATE["tesseract"] = TesseractEngine(cfg.ocr.engine_b, models_dir(cfg.runtime.models_dir))
    return _STATE["tesseract"]


def _faces() -> Any:
    if "faces" not in _STATE:
        from mlredact.layout.faces import FaceDetector

        cfg = _config()
        paths = resolve(cfg.regions.face_model, models_dir(cfg.runtime.models_dir))
        model = paths[next(name for name in sorted(paths) if name.endswith(".onnx"))]
        _STATE["faces"] = FaceDetector(model, cfg.regions.face_score_threshold)
    return _STATE["faces"]


def _visual_regions(rgb: Any, page_index: int, dpi: int) -> list[Region]:
    cfg = _config().regions
    found: list[Region] = []
    if cfg.barcodes:
        found.extend(find_barcodes(rgb, page_index, dpi))
    if cfg.faces:
        found.extend(_faces().find(rgb, page_index))
    return found


def _without(lines: tuple[Line, ...], token_ids: frozenset[str]) -> tuple[Line, ...]:
    if not token_ids:
        return lines
    out = []
    for line in lines:
        kept = tuple(t for t in line.tokens if t.token_id not in token_ids)
        if kept:
            out.append(replace(line, tokens=kept, bbox=union_all([t.bbox for t in kept])))
    return tuple(out)


def task_analyse(ref: BlobRef, page_index: int, ocr_profile: str = "primary") -> PageAnalysis:
    """Render + OCR + regions.  The primary pass also accounts for every patch of ink (re-reading
    what OCR missed); the verification pass (over the *output*) looks for surviving barcodes/faces."""
    try:
        cfg = _config()
        primary = ocr_profile == "primary"
        rendered = render_page(
            _document(ref), page_index, cfg.render, cfg.limits, with_words=primary and cfg.ocr.embedded_text
        )
        engine = _engine(ocr_profile)
        lines = engine.ocr_page(rendered.rgb, page_index)
        if rendered.words:
            lines = fuse_embedded(
                lines, rendered.words, rendered.rgb, page_index, replace_below=cfg.ocr.embedded_replace_below
            )
        if cfg.ocr.engine_b.enabled:  # both passes: V4 re-reads the output with A and B (plan §12)
            lines = fuse_tesseract(
                lines,
                _tesseract().words(rendered.rgb, rendered.geometry.dpi),
                page_index,
                replace_below=cfg.ocr.embedded_replace_below,
                min_new_confidence=cfg.ocr.engine_b.min_new_word_confidence / 100.0,
            )
        regions: tuple[Region, ...] = ()
        unexplained = 0.0
        if cfg.regions.enabled:
            dpi = rendered.geometry.dpi
            visual = _visual_regions(rendered.rgb, page_index, dpi)
            if primary:
                found = analyse_regions(
                    rendered.rgb,
                    lines,
                    page_index,
                    dpi,
                    cfg.regions,
                    detected=visual,
                    reread=lambda crop: engine.ocr_page(crop, page_index),
                )
                if found.illegible and cfg.regions.illegible_action == "quarantine":
                    raise ProcessingError(ReasonCode.PAGE_ILLEGIBLE, page=page_index)
                lines = _without(lines, found.ghost_tokens) + found.extra_lines
                regions, unexplained = found.regions, found.unexplained_ink
            else:
                regions = tuple(visual)
        return PageAnalysis(
            geometry=rendered.geometry,
            kind=_page_kind(rendered.embedded_char_count, rendered.image_object_count),
            lines=lines,
            embedded_char_count=rendered.embedded_char_count,
            regions=regions,
            unexplained_ink=unexplained,
        )
    except MlredactError:
        raise
    except Exception:
        raise _guard(ReasonCode.OCR_FAILED, page_index) from None


@dataclass(frozen=True, slots=True)
class RedactedPage:
    geometry: PageGeometry
    image: EncodedImage
    fills_ok: bool
    ops_applied: int
    applied: tuple[Applied, ...] = ()
    notice_box: BBox | None = None


def task_redact(ref: BlobRef, page_index: int, ops: tuple[RenderOp, ...], notice: str | None = None) -> RedactedPage:
    try:
        cfg = _config()
        rendered = render_page(_document(ref), page_index, cfg.render, cfg.limits)
        edit = apply_ops(
            rendered.rgb,
            ops,
            cfg.redaction,
            cfg.surrogate.font_family,
            notice=notice,
            dpi=rendered.geometry.dpi,
        )
        return RedactedPage(
            rendered.geometry,
            encode_raster(edit.raster, cfg.output),
            edit.verified,
            len(ops),
            edit.applied,
            edit.notice_box,
        )
    except MlredactError:
        raise
    except Exception:
        raise _guard(ReasonCode.RENDER_FAILED, page_index) from None


def models_root() -> Path:
    return models_dir(_config().runtime.models_dir)
