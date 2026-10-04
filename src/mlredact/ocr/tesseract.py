"""Engine B: Tesseract 5 (LSTM), an independent second reader (plan §6).

Runs the system binary as a subprocess: the page goes in as PNG on stdin, words come back as TSV
with boxes and confidences.  Determinism: one OpenMP thread, fixed DPI (no resolution estimation),
a hash-pinned ``traineddata`` model and the binary's version recorded in the runtime profile.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from mlredact.config.schema import OcrEngineBConfig
from mlredact.core.errors import EnvironmentErrorMl, ProcessingError, ReasonCode
from mlredact.core.geometry import BBox
from mlredact.registry.models import resolve

ENGINE_ID = "tesseract"


@dataclass(frozen=True, slots=True)
class ReaderWord:
    """A word from a second reader (engine B or D) in canonical pixels."""

    text: str
    bbox: BBox
    confidence: float  # 0..1
    line: int  # reader-specific line grouping key


def binary_version(binary: str = "tesseract") -> str | None:
    path = shutil.which(binary)
    if path is None:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30, check=False)  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return None
    first = (out.stdout or out.stderr).strip().splitlines()
    return first[0].strip() if first else None


def parse_tsv(tsv: str) -> list[ReaderWord]:
    """Tesseract TSV -> words (level 5 rows), grouped by (block, paragraph, line) in output order."""
    words: list[ReaderWord] = []
    keys: dict[tuple[int, int, int], int] = {}
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        text = cols[11].strip()
        conf = float(cols[10])
        if not text or conf < 0:
            continue
        left, top, width, height = (int(v) for v in cols[6:10])
        if width <= 0 or height <= 0:
            continue
        key = (int(cols[2]), int(cols[3]), int(cols[4]))
        line = keys.setdefault(key, len(keys))
        words.append(ReaderWord(text, BBox(left, top, left + width, top + height), round(conf / 100.0, 4), line))
    return words


class TesseractEngine:
    engine_id = ENGINE_ID

    def __init__(self, cfg: OcrEngineBConfig, models_dir: Path) -> None:
        binary = shutil.which(cfg.binary)
        version = binary_version(cfg.binary)
        if binary is None or version is None:
            raise EnvironmentErrorMl(ReasonCode.MODEL_MISSING)
        paths = resolve(cfg.model, models_dir)
        self._binary = binary
        self._tessdata = str(next(iter(paths.values())).parent)
        self._cfg = cfg
        self.version = version

    def words(self, rgb: npt.NDArray[np.uint8], dpi: int) -> list[ReaderWord]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ok, png = cv2.imencode(".png", gray)
        if not ok:
            raise ProcessingError(ReasonCode.OCR_FAILED)
        cmd = [
            self._binary,
            "stdin",
            "stdout",
            "--tessdata-dir",
            self._tessdata,
            "-l",
            self._cfg.language,
            "--oem",
            "1",
            "--psm",
            str(self._cfg.page_segmentation),
            "--dpi",
            str(dpi),
            "tsv",
        ]
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        try:
            out = subprocess.run(  # noqa: S603 - fixed binary and arguments, no shell
                cmd, input=png.tobytes(), capture_output=True, timeout=self._cfg.timeout_s, check=False, env=env
            )
        except (OSError, subprocess.SubprocessError):
            raise ProcessingError(ReasonCode.OCR_FAILED) from None
        if out.returncode != 0:
            raise ProcessingError(ReasonCode.OCR_FAILED)
        return parse_tsv(out.stdout.decode("utf-8", errors="replace"))
