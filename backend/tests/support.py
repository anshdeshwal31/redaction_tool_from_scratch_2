"""Test helpers: configuration for the current machine, and synthetic PageAnalysis objects (no OCR)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlredact.config.loader import load_config
from mlredact.config.schema import AppConfig
from mlredact.core.geometry import BBox
from mlredact.core.model import Line, PageAnalysis, PageGeometry, Token, token_id
from mlredact.core.types import PageKind

# Machine-specific deployment files layered into every test configuration, e.g. on a laptop without
# the NER models:  MLREDACT_TEST_CONFIG=~/.mlredact/laptop-no-ner.yaml  (os.pathsep-separated list).
EXTRA_CONFIG = tuple(
    Path(p).expanduser() for p in os.environ.get("MLREDACT_TEST_CONFIG", "").split(os.pathsep) if p.strip()
)


def make_config(profile: str = "broad", overrides: Mapping[str, Any] | None = None) -> AppConfig:
    return load_config(profile, EXTRA_CONFIG, overrides)


def ner_enabled() -> bool:
    ner = make_config().detection.ner
    return ner.gliner.enabled or ner.privacy_filter.enabled


def pages_from_lines(
    lines: Sequence[str],
    *,
    confidences: dict[str, float] | None = None,
    alternates: dict[str, tuple[str, ...]] | None = None,
) -> list[PageAnalysis]:
    confidences = confidences or {}
    alternates = alternates or {}
    geometry = PageGeometry(0, 595.276, 841.89, 300, 2481, 3508)
    out_lines: list[Line] = []
    row = 0  # a blank input line leaves a vertical gap (paragraph break) but creates no Line
    line_no = -1
    for text in lines:
        row += 1
        if not text.strip():
            continue
        line_no += 1
        tokens: list[Token] = []
        x = 100
        y = 100 + (row - 1) * 60
        for word_no, word in enumerate(text.split()):
            w = 25 * len(word)
            conf = confidences.get(word, 0.99)
            tokens.append(
                Token(
                    token_id=token_id(0, line_no, word_no),
                    page=0,
                    line_no=line_no,
                    word_no=word_no,
                    bbox=BBox(x, y, x + w, y + 50),
                    confidence=conf,
                    min_confidence=conf,
                    engine="test",
                    text=word,
                    alternates=alternates.get(word, ()),
                )
            )
            x += w + 20
        out_lines.append(
            Line(
                line_id=f"p0000.l{line_no:04d}",
                page=0,
                line_no=line_no,
                bbox=BBox(100, y, max(x, 101), y + 50),
                quad=((100.0, y), (x, y), (x, y + 50.0), (100.0, y + 50.0)),
                confidence=0.99,
                engine="test",
                tokens=tuple(tokens),
            )
        )
    return [PageAnalysis(geometry, PageKind.IMAGE_ONLY, tuple(out_lines))]


# ------------------------------------------------------------- worker-pool tasks (importable by spawned children)
def task_echo(value: int) -> int:
    return value * 2


def task_reject(page: int) -> None:
    from mlredact.core.errors import InputRejected, ReasonCode

    raise InputRejected(ReasonCode.INPUT_ENCRYPTED, page=page)


def task_crash() -> None:
    import os

    os._exit(3)  # simulates a native crash (e.g. a hostile PDF segfaulting a parser)
