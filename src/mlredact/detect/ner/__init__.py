"""Statistical NER detectors (plan §7.2 N1/N2), built once per :class:`~mlredact.pipeline.runner.Redactor`."""

from __future__ import annotations

from pathlib import Path

from mlredact.config.schema import AppConfig
from mlredact.detect.base import Detector


def ner_model_ids(config: AppConfig) -> list[str]:
    ner = config.detection.ner
    ids = []
    if ner.gliner.enabled:
        ids.append(ner.gliner.model)
    if ner.privacy_filter.enabled:
        ids.append(ner.privacy_filter.model)
    return ids


def build_ner_detectors(config: AppConfig, models_dir: Path) -> tuple[Detector, ...]:
    """Load the enabled NER models (hash-verified).  Order is fixed: N1 then N2."""
    ner = config.detection.ner
    out: list[Detector] = []
    if ner.privacy_filter.enabled:
        from mlredact.detect.ner.privacy_filter import PrivacyFilterDetector

        out.append(PrivacyFilterDetector(ner.privacy_filter, ner.threads, models_dir))
    if ner.gliner.enabled:
        from mlredact.detect.ner.gliner import GlinerDetector

        out.append(GlinerDetector(ner.gliner, ner.threads, models_dir))
    return tuple(out)
