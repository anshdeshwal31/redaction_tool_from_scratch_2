"""Face detector smoke test (needs the pinned YuNet model)."""

from __future__ import annotations

import numpy as np
import pytest

from mlredact.config.loader import load_config
from mlredact.layout.faces import FaceDetector
from mlredact.registry.models import models_dir, resolve

pytestmark = [pytest.mark.slow, pytest.mark.models]


def test_yunet_loads_and_finds_nothing_on_a_text_page() -> None:
    cfg = load_config("broad")
    paths = resolve(cfg.regions.face_model, models_dir(cfg.runtime.models_dir))
    detector = FaceDetector(
        next(p for n, p in sorted(paths.items()) if n.endswith(".onnx")), cfg.regions.face_score_threshold
    )
    page = np.full((3508, 2481, 3), 250, dtype=np.uint8)
    page[300:340, 200:2000] = 20  # a "line of text"
    assert detector.find(page, 0) == []
    assert detector.find(page, 0) == detector.find(page.copy(), 0)  # deterministic
