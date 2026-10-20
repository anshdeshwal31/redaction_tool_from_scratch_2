"""Originals kept for side-by-side review, for a job that asks for it.

A PDF is kept as uploaded.  A scan (TIFF, PNG, JPEG) is kept as the image-only PDF the engine builds
from it, so the browser shows both sides with one PDF reader and their pages line up.  Decoding an
untrusted image runs in its own process, as it does in the engine.

    python -m mlredact.api.originals <image> <pdf> <max pages> <max page pixels>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mlredact.config.schema import LimitsConfig
from mlredact.ingest.images import image_to_pdf, sniff_format

CONVERT_TIMEOUT_S = 120.0
_SCANS = frozenset({"tiff", "png", "jpeg"})


def keep(source: Path, target: Path, limits: LimitsConfig) -> bool:
    """Move ``source`` to ``target`` as a PDF.  False when it is neither a PDF nor a readable scan."""
    with source.open("rb") as f:
        fmt = sniff_format(f.read(1024))
    if fmt == "pdf":
        source.replace(target)  # moved, never copied: one copy of the document at any time
        return True
    if fmt not in _SCANS:
        return False
    part = target.with_name(target.name + ".part")
    args = [str(source), str(part), str(limits.max_pages), str(limits.max_page_pixels)]
    try:
        done = subprocess.run(  # noqa: S603 - this module, with this Python, no shell
            [sys.executable, "-m", __name__, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=CONVERT_TIMEOUT_S,
            check=False,
        )
        if done.returncode != 0 or not part.is_file():
            return False
        part.replace(target)
        return True
    except subprocess.TimeoutExpired:
        return False
    finally:
        part.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    source, target, max_pages, max_page_pixels = argv
    limits = LimitsConfig(max_pages=int(max_pages), max_page_pixels=int(max_page_pixels))
    try:
        pdf = image_to_pdf(Path(source).read_bytes(), limits)
    except Exception:
        return 1  # no message: it could carry details of the document
    Path(target).write_bytes(pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
