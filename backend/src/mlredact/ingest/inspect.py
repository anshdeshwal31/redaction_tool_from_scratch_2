"""Structural inspection of an input PDF (runs inside a worker process, never in the parent).

The report records what will be *dropped* by rebuild-based redaction (attachments, scripts,
annotation popups, hidden layers) so it can be surfaced in the manifest, and rejects inputs that
cannot be rendered faithfully (password-protected, dynamic-XFA-only, over limits).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from mlredact.config.schema import LimitsConfig
from mlredact.core.errors import InputRejected, ReasonCode


@dataclass(frozen=True, slots=True)
class InspectionReport:
    page_count: int
    encrypted: bool  # e.g. owner-password restrictions; content is still renderable
    has_acroform: bool
    has_xfa: bool
    has_javascript: bool
    attachment_count: int
    annotation_count: int
    has_optional_content: bool
    user_unit_pages: int
    # Each page's /Rotate as the renderer applies it: 0, 90, 180 or 270 (clockwise).  Not in the manifest: it only
    # tells why a sideways page is held (layout.orientation).
    rotations: tuple[int, ...] = ()

    def as_manifest(self) -> dict[str, object]:
        return {
            "page_count": self.page_count,
            "encrypted": self.encrypted,
            "has_acroform": self.has_acroform,
            "has_xfa": self.has_xfa,
            "has_javascript": self.has_javascript,
            "attachments_dropped": self.attachment_count,
            "annotations_flattened_or_dropped": self.annotation_count,
            "has_optional_content": self.has_optional_content,
            "user_unit_pages": self.user_unit_pages,
        }


def inspect_pdf(data: bytes, limits: LimitsConfig) -> InspectionReport:
    import pikepdf

    if not data:
        raise InputRejected(ReasonCode.INPUT_EMPTY)
    if len(data) > limits.max_input_bytes:
        raise InputRejected(ReasonCode.INPUT_TOO_LARGE)
    if not data.lstrip()[:1024].startswith(b"%PDF") and b"%PDF-" not in data[:1024]:
        raise InputRejected(ReasonCode.INPUT_UNSUPPORTED)
    try:
        pdf = pikepdf.open(io.BytesIO(data))
    except pikepdf.PasswordError:
        raise InputRejected(ReasonCode.INPUT_ENCRYPTED) from None
    except Exception:
        raise InputRejected(ReasonCode.INPUT_UNREADABLE) from None
    with pdf:
        try:
            pages = len(pdf.pages)
        except Exception:
            raise InputRejected(ReasonCode.INPUT_UNREADABLE) from None
        if pages == 0:
            raise InputRejected(ReasonCode.INPUT_EMPTY)
        if pages > limits.max_pages:
            raise InputRejected(ReasonCode.INPUT_TOO_MANY_PAGES)
        root = pdf.Root
        acro = root.get("/AcroForm")
        has_acroform = acro is not None
        has_xfa = bool(acro is not None and "/XFA" in acro)
        needs_rendering = bool(root.get("/NeedsRendering", False))
        names = root.get("/Names")
        has_js = bool((names is not None and "/JavaScript" in names) or "/OpenAction" in root or "/AA" in root)
        annotations = 0
        user_unit = 0
        rotations: list[int] = []
        for page in pdf.pages:
            rotations.append(_rotation(page.obj.get("/Rotate")))
            annots = page.obj.get("/Annots")
            if annots is not None:
                try:
                    annotations += len(annots)
                except Exception:
                    annotations += 1
            if "/UserUnit" in page.obj:
                user_unit += 1
        # Dynamic XFA forms have no static page content PDFium (non-XFA build) can render.
        if has_xfa and needs_rendering:
            raise InputRejected(ReasonCode.INPUT_XFA_ONLY)
        return InspectionReport(
            page_count=pages,
            encrypted=bool(pdf.is_encrypted),
            has_acroform=has_acroform,
            has_xfa=has_xfa,
            has_javascript=has_js,
            attachment_count=len(pdf.attachments),
            annotation_count=annotations,
            has_optional_content="/OCProperties" in root,
            user_unit_pages=user_unit,
            rotations=tuple(rotations),
        )


def _rotation(value: object) -> int:
    """A page's /Rotate (pikepdf's page list carries an inherited one down to the page) as PDFium applies it: whole
    quarter turns counted toward zero, so 450 is 90, -90 is 270 and 45 is 0; a missing or malformed value is 0."""
    try:
        degrees = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0
    return (int(degrees / 90) % 4) * 90
