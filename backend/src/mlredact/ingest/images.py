"""Image inputs (TIFF, PNG, JPEG) -> a deterministic image-only PDF (Appendix A "input").

Runs in a worker process (decoding untrusted images carries the same risk as parsing PDFs).  Every
frame becomes one page at the image's own resolution; JPEG data is embedded unchanged and other
images are stored losslessly (Flate), so the canonical raster sees exactly the scanned pixels.
"""

from __future__ import annotations

import io
import zlib
from collections.abc import Mapping
from typing import Any

from mlredact.config.schema import LimitsConfig
from mlredact.core.errors import InputRejected, ReasonCode

_DEFAULT_DPI = 300.0


def sniff_format(data: bytes) -> str | None:
    head = data[:16]
    if data.lstrip()[:1024].startswith(b"%PDF") or b"%PDF-" in data[:1024]:
        return "pdf"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def _dpi(info: Mapping[Any, Any]) -> float:
    raw: Any = info.get("dpi")
    try:
        x = float(raw[0]) if isinstance(raw, tuple) else float(raw)
    except (TypeError, ValueError, IndexError):
        return _DEFAULT_DPI
    return x if 72.0 <= x <= 1200.0 else _DEFAULT_DPI  # missing / absurd metadata: assume a 300 dpi scan


def image_to_pdf(data: bytes, limits: LimitsConfig) -> bytes:
    import pikepdf
    from PIL import Image

    fmt = sniff_format(data)
    if fmt not in {"tiff", "png", "jpeg"}:
        raise InputRejected(ReasonCode.INPUT_UNSUPPORTED)
    Image.MAX_IMAGE_PIXELS = limits.max_page_pixels  # decompression-bomb guard (raises on excess)
    try:
        src = Image.open(io.BytesIO(data))
        n_frames = int(getattr(src, "n_frames", 1))
    except Exception:
        raise InputRejected(ReasonCode.INPUT_UNREADABLE) from None
    if n_frames < 1:
        raise InputRejected(ReasonCode.INPUT_EMPTY)
    if n_frames > limits.max_pages:
        raise InputRejected(ReasonCode.INPUT_TOO_MANY_PAGES)
    pdf = pikepdf.new()
    for index in range(n_frames):
        # One frame at a time: ``ImageSequence`` yields the *same* object re-seeked, so a list of
        # frames would hold N references to the last page.
        try:
            src.seek(index)
            frame = src.copy()
            frame.info = dict(src.info)
        except Exception:
            raise InputRejected(ReasonCode.INPUT_UNREADABLE) from None
        try:
            dpi = _dpi(frame.info)
            w, h = frame.size
            if w * h > limits.max_page_pixels:
                raise InputRejected(ReasonCode.INPUT_TOO_LARGE)
            if fmt == "jpeg" and frame.mode in {"L", "RGB"}:
                stream = pikepdf.Stream(pdf, data)  # the original JPEG bytes, unchanged
                stream.Filter = pikepdf.Name.DCTDecode
                mode = frame.mode
            else:
                if frame.mode in {"L", "RGB"}:
                    img = frame.copy()
                elif frame.mode in {"1", "I", "I;16", "I;16B", "F"}:  # bilevel fax scans, 16-bit grey
                    img = frame.convert("L")
                else:  # palette, CMYK, alpha: flatten onto white paper
                    rgba = frame.convert("RGBA")
                    img = Image.new("RGB", frame.size, (255, 255, 255))
                    img.paste(rgba, mask=rgba.getchannel("A"))
                mode = img.mode
                stream = pikepdf.Stream(pdf, zlib.compress(img.tobytes(), 6))
                stream.Filter = pikepdf.Name.FlateDecode
        except InputRejected:
            raise
        except Exception:
            raise InputRejected(ReasonCode.INPUT_UNREADABLE) from None
        stream.Type = pikepdf.Name.XObject
        stream.Subtype = pikepdf.Name.Image
        stream.Width, stream.Height = w, h
        stream.ColorSpace = pikepdf.Name.DeviceGray if mode == "L" else pikepdf.Name.DeviceRGB
        stream.BitsPerComponent = 8
        w_pt, h_pt = w * 72.0 / dpi, h * 72.0 / dpi
        content = pikepdf.Stream(pdf, f"q {w_pt:.4f} 0 0 {h_pt:.4f} 0 0 cm /Im0 Do Q".encode())
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page,
            MediaBox=[0, 0, round(w_pt, 4), round(h_pt, 4)],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=stream)),
            Contents=content,
        )
        pdf.pages.append(pikepdf.Page(page))
    out = io.BytesIO()
    pdf.save(out, deterministic_id=True, compress_streams=False)
    return out.getvalue()
