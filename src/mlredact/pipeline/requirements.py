"""The pinned models a configuration needs (single source of truth for the runner, selfcheck and fetch)."""

from __future__ import annotations

from mlredact.config.schema import AppConfig, OcrEngineAConfig
from mlredact.detect.ner import ner_model_ids


def _ocr(cfg: OcrEngineAConfig) -> list[str]:
    return [cfg.det_model, cfg.rec_model, *([cfg.cls_model] if cfg.use_textline_orientation else [])]


def required_model_ids(config: AppConfig) -> list[str]:
    ids = {*_ocr(config.ocr.engine_a), *ner_model_ids(config)}
    if config.verification.re_ocr:
        ids.update(_ocr(config.verification.ocr))
    if config.regions.enabled and config.regions.faces:
        ids.add(config.regions.face_model)
    if config.ocr.engine_b.enabled:
        ids.add(config.ocr.engine_b.model)
    return sorted(ids)
