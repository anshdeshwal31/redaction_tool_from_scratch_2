"""Build the output PDF from scratch (plan §11).

Nothing is copied from the input: every page is one image XObject drawn full-page, plus an optional
invisible text layer generated only from post-redaction content.  Metadata is fixed, the /ID is
derived from content and object order is deterministic, so identical inputs give identical bytes.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

import pikepdf

from mlredact import __version__
from mlredact.config.schema import OutputConfig
from mlredact.pdfout.encode import EncodedImage

TITLE = "De-identified document"


@dataclass(frozen=True, slots=True)
class PageSpec:
    width_pt: float
    height_pt: float
    image: EncodedImage
    text_layer: bytes | None = None  # content-stream fragment (BT..ET), already invisible (Tr 3)


def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def build_pdf(pages: Sequence[PageSpec], cfg: OutputConfig) -> bytes:
    pdf = pikepdf.new()
    font_resources: pikepdf.Dictionary | None = None
    if any(spec.text_layer is not None for spec in pages):
        helvetica = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font,
                Subtype=pikepdf.Name.Type1,
                BaseFont=pikepdf.Name.Helvetica,
                Encoding=pikepdf.Name.WinAnsiEncoding,
            )
        )
        font_resources = pikepdf.Dictionary(F1=helvetica)
    for spec in pages:
        img = spec.image
        xobj = pikepdf.Stream(pdf, img.data)
        xobj.Type = pikepdf.Name.XObject
        xobj.Subtype = pikepdf.Name.Image
        xobj.Width = img.width
        xobj.Height = img.height
        xobj.ColorSpace = pikepdf.Name("/" + img.colorspace)
        xobj.BitsPerComponent = 8
        xobj.Filter = pikepdf.Name("/" + img.filter)
        w, h = _fmt(spec.width_pt), _fmt(spec.height_pt)
        content = f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q\n".encode("ascii")
        resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=xobj))
        if spec.text_layer is not None:
            content += spec.text_layer
            if font_resources is not None:
                resources.Font = font_resources
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page,
            MediaBox=[0, 0, float(w), float(h)],
            Resources=resources,
            Contents=pikepdf.Stream(pdf, content),
        )
        pdf.pages.append(pikepdf.Page(page))
    pdf.docinfo = pdf.make_indirect(
        pikepdf.Dictionary(Producer=pikepdf.String(f"{cfg.producer} {__version__}"), Title=pikepdf.String(TITLE))
    )
    out = io.BytesIO()
    pdf.save(
        out,
        deterministic_id=True,
        compress_streams=True,
        object_stream_mode=pikepdf.ObjectStreamMode.disable,
        normalize_content=False,
        linearize=False,
        fix_metadata_version=False,
    )
    return out.getvalue()


def build_text_pdf(pages: Sequence[tuple[float, float, bytes | None]]) -> bytes:
    """A PDF of text layers alone: (width, height, layer) per page, no images, no metadata.  Verification reads the
    exempt rendition of an output's text layer from it with the extraction it reads the output with; it is never
    released."""
    pdf = pikepdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
            Encoding=pikepdf.Name.WinAnsiEncoding,
        )
    )
    for width, height, layer in pages:
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page,
            MediaBox=[0, 0, float(_fmt(width)), float(_fmt(height))],
            Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)),
            Contents=pikepdf.Stream(pdf, layer or b""),
        )
        pdf.pages.append(pikepdf.Page(page))
    out = io.BytesIO()
    pdf.save(out, deterministic_id=True)
    return out.getvalue()
