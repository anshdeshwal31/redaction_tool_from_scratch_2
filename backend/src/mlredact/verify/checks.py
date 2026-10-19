"""Release-gate checks V1, V2 and V6 (plan §12).

This module deliberately imports nothing from ``mlredact.render`` or ``mlredact.detect`` internals:
it parses the *output bytes* independently, so a rendering bug cannot hide itself.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from mlredact.core.types import VerificationCheck
from mlredact.text.normalize import digits_only, match_key


@dataclass(frozen=True, slots=True)
class CheckResult:
    check: VerificationCheck
    passed: bool
    metrics: dict[str, int | float | bool] = field(default_factory=dict)

    def manifest(self) -> dict[str, object]:
        return {"check": self.check.value, "passed": self.passed, "metrics": dict(sorted(self.metrics.items()))}


# ---------------------------------------------------------------------------------------- V1
_ALLOWED_TRAILER = {"/Root", "/Info", "/ID", "/Size"}
_ALLOWED_CATALOG = {"/Type", "/Pages"}
_ALLOWED_PAGES = {"/Type", "/Kids", "/Count"}
_ALLOWED_PAGE = {"/Type", "/Parent", "/MediaBox", "/Resources", "/Contents"}
_ALLOWED_RESOURCES = {"/XObject", "/Font"}
_ALLOWED_INFO = {"/Producer", "/Title"}
_ALLOWED_FONT = {"/Type", "/Subtype", "/BaseFont", "/Encoding"}
_ALLOWED_IMAGE = {
    "/Type",
    "/Subtype",
    "/Width",
    "/Height",
    "/ColorSpace",
    "/BitsPerComponent",
    "/Filter",
    "/Length",
    "/DecodeParms",
}


def check_structure(pdf_bytes: bytes, expected_pages: int, *, allow_fonts: bool) -> CheckResult:
    """V1: the object graph contains only what the builder is allowed to emit."""
    import pikepdf

    problems = 0
    metrics: dict[str, int | float | bool] = {}
    # Single revision: exactly one cross-reference section / EOF marker.
    metrics["eof_markers"] = pdf_bytes.count(b"%%EOF")
    metrics["startxref_markers"] = pdf_bytes.count(b"startxref")
    if metrics["eof_markers"] != 1 or metrics["startxref_markers"] != 1:
        problems += 1
    try:
        pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    except Exception:  # an output we cannot even parse is a structural failure, not a crash
        metrics["unparseable"] = True
        metrics["problems"] = problems + 1
        return CheckResult(VerificationCheck.STRUCTURE, False, metrics)
    with pdf:
        trailer_keys = set(pdf.trailer.keys())
        problems += len(trailer_keys - _ALLOWED_TRAILER)
        root = pdf.Root
        problems += len(set(root.keys()) - _ALLOWED_CATALOG)
        problems += len(set(root.Pages.keys()) - _ALLOWED_PAGES)
        info = pdf.trailer.get("/Info")
        if info is not None:
            problems += len(set(info.keys()) - _ALLOWED_INFO)
        metrics["pages"] = len(pdf.pages)
        if len(pdf.pages) != expected_pages:
            problems += 1
        images = 0
        for page in pdf.pages:
            obj = page.obj
            problems += len(set(obj.keys()) - _ALLOWED_PAGE)
            res = obj.get("/Resources", pikepdf.Dictionary())
            problems += len(set(res.keys()) - _ALLOWED_RESOURCES)
            if "/Font" in res:
                if not allow_fonts:
                    problems += 1
                else:
                    for name, font in res.Font.items():
                        if (
                            name != "/F1"
                            or set(font.keys()) - _ALLOWED_FONT
                            or (
                                font.get("/Subtype") != pikepdf.Name.Type1
                                or font.get("/BaseFont") != pikepdf.Name.Helvetica
                            )
                        ):
                            problems += 1
            xobjects = res.get("/XObject", pikepdf.Dictionary())
            for _name, x in xobjects.items():
                if x.get("/Subtype") != pikepdf.Name.Image:
                    problems += 1
                problems += len(set(x.keys()) - _ALLOWED_IMAGE)
                if "/SMask" in x or "/Mask" in x or "/Alternates" in x:
                    problems += 1
                images += 1
        metrics["images"] = images
        if images != expected_pages:
            problems += 1
    # The output must also open and render-count correctly in an independent engine.
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            metrics["pdfium_pages"] = len(doc)
            if len(doc) != expected_pages:
                problems += 1
        finally:
            doc.close()
    except Exception:
        problems += 1
    metrics["problems"] = problems
    return CheckResult(VerificationCheck.STRUCTURE, problems == 0, metrics)


# ---------------------------------------------------------------------------------------- V2
def _value_needles(values: Iterable[str], min_len: int) -> list[str]:
    needles: set[str] = set()
    for v in values:
        key = match_key(v)
        if len(key) >= min_len:
            needles.add(key)
        digits = digits_only(v)
        if len(digits) >= max(min_len, 6):
            needles.add(digits)
    return sorted(needles)


def _decode_literal(body: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == BACKSLASH and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in "01234567":
                j = i + 1
                while j < len(body) and j < i + 4 and body[j] in "01234567":
                    j += 1
                out.append(chr(int(body[i + 1 : j], 8)))
                i = j
                continue
            out.append({"n": " ", "r": " ", "t": " ", "b": "", "f": ""}.get(nxt, nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


BACKSLASH = chr(92)
_LITERAL = re.compile(r"[(]((?:[^()" + BACKSLASH * 2 + r"]|" + BACKSLASH * 2 + r".)*)[)]")
_HEXSTR = re.compile(r"<([0-9A-Fa-f" + BACKSLASH + r"s]+)>")


def text_segments(pdf_bytes: bytes) -> list[str]:
    """Every textual payload in the file, as separate segments: PDFium-extracted lines, the string
    operands of every non-image stream, and every string object / info value."""
    import pikepdf
    import pypdfium2 as pdfium

    segments: list[str] = []
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            try:
                segments.extend(tp.get_text_range().splitlines())
            finally:
                tp.close()
                page.close()
    finally:
        doc.close()
    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        for obj in pdf.objects:
            if isinstance(obj, pikepdf.Stream):
                if obj.get("/Subtype") == pikepdf.Name.Image:
                    continue
                try:
                    raw = obj.read_bytes().decode("latin-1")
                except Exception:
                    raw = obj.read_raw_bytes().decode("latin-1")
                segments.extend(_decode_literal(m.group(1)) for m in _LITERAL.finditer(raw))
                for m in _HEXSTR.finditer(raw):
                    hexdigits = "".join(m.group(1).split())
                    if len(hexdigits) % 2 == 0:
                        segments.append(bytes.fromhex(hexdigits).decode("latin-1"))
            elif isinstance(obj, pikepdf.String):
                segments.append(str(obj))
        info = pdf.trailer.get("/Info")
        if info is not None:
            segments.extend(str(v) for v in info.values())
    return [seg for seg in segments if seg.strip()]


def extract_all_text(pdf_bytes: bytes) -> str:
    return chr(10).join(text_segments(pdf_bytes))


def check_text_leak(pdf_bytes: bytes, removed_values: Sequence[str], *, min_len: int = 4) -> CheckResult:
    """V2: no removed value (or its digit sequence) is present in any extractable text segment."""
    segments = text_segments(pdf_bytes)
    keys = [match_key(seg) for seg in segments]
    digit_runs = [digits_only(seg) for seg in segments]
    needles = _value_needles(removed_values, min_len)
    hits = 0
    for n in needles:
        pool = digit_runs if n.isdigit() else keys
        if any(n in hay for hay in pool):
            hits += 1
    return CheckResult(
        VerificationCheck.TEXT,
        hits == 0,
        {"needles": len(needles), "hits": hits, "segments": len(segments), "text_chars": sum(map(len, segments))},
    )


# ---------------------------------------------------------------------------------------- V6
def check_integrity(
    expected_sizes: Sequence[tuple[float, float]],
    pdf_bytes: bytes,
    planned_ops: int,
    applied_ops: int,
    fills_ok: Sequence[bool],
) -> tuple[CheckResult, CheckResult]:
    """V6 integrity and the V3 (pre-encoding pixel) summary."""
    import pikepdf

    problems = 0
    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        if len(pdf.pages) != len(expected_sizes):
            problems += 1
        else:
            for page, (w, h) in zip(pdf.pages, expected_sizes, strict=True):
                box = [float(v) for v in page.obj.MediaBox]
                if abs(box[2] - box[0] - w) > 0.01 or abs(box[3] - box[1] - h) > 0.01:
                    problems += 1
    if planned_ops != applied_ops:
        problems += 1
    integrity = CheckResult(
        VerificationCheck.INTEGRITY,
        problems == 0,
        {"problems": problems, "planned_ops": planned_ops, "applied_ops": applied_ops},
    )
    pixels = CheckResult(
        VerificationCheck.PIXELS,
        all(fills_ok),
        {"pages_checked": len(fills_ok), "pages_failed": sum(not f for f in fills_ok)},
    )
    return integrity, pixels
