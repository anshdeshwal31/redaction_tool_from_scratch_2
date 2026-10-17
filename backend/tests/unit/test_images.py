"""Image inputs (TIFF / PNG / JPEG) become deterministic image-only PDFs."""

from __future__ import annotations

import io

import pikepdf
import pytest
from PIL import Image, ImageDraw

from mlredact.config.schema import LimitsConfig
from mlredact.core.errors import InputRejected, ReasonCode
from mlredact.ingest.images import image_to_pdf, sniff_format


def _encode(img: Image.Image, fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _page(mode: str = "L", size: tuple[int, int] = (850, 1100)) -> Image.Image:
    img = Image.new(mode, size, "white" if mode != "1" else 1)
    ImageDraw.Draw(img).text((60, 60), "Patient: Zelda QUIMBY", fill="black" if mode != "1" else 0)
    return img


def test_formats_are_sniffed_from_content_not_names() -> None:
    assert sniff_format(b"%PDF-1.7\n...") == "pdf"
    assert sniff_format(_encode(_page(), "TIFF")) == "tiff"
    assert sniff_format(_encode(_page(), "PNG")) == "png"
    assert sniff_format(_encode(_page(), "JPEG")) == "jpeg"
    assert sniff_format(b"GIF89a....") is None


def test_multipage_tiff_keeps_resolution_and_is_deterministic() -> None:
    fax, colour = _page("1"), _page("RGB")
    data = _encode(fax, "TIFF", save_all=True, append_images=[colour], dpi=(200, 200))
    a, b = image_to_pdf(data, LimitsConfig()), image_to_pdf(data, LimitsConfig())
    assert a == b
    with pikepdf.open(io.BytesIO(a)) as pdf:
        assert len(pdf.pages) == 2
        assert [float(v) for v in pdf.pages[0].obj.MediaBox] == [0, 0, 306.0, 396.0]  # 850 x 1100 px at 200 dpi
        spaces = [str(next(iter(p.get_images().values())).ColorSpace) for p in pdf.pages]
        assert spaces == ["/DeviceGray", "/DeviceRGB"]


def test_jpeg_bytes_are_embedded_unchanged_and_alpha_is_flattened() -> None:
    jpeg = _encode(_page("RGB"), "JPEG", quality=80)
    with pikepdf.open(io.BytesIO(image_to_pdf(jpeg, LimitsConfig()))) as pdf:
        (image,) = pdf.pages[0].get_images().values()
        assert image.read_raw_bytes() == jpeg
    rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    with pikepdf.open(io.BytesIO(image_to_pdf(_encode(rgba, "PNG"), LimitsConfig()))) as pdf:
        (image,) = pdf.pages[0].get_images().values()
        assert set(image.read_bytes()) == {255}  # transparent -> white paper


def test_oversized_and_corrupt_images_are_rejected() -> None:
    small = LimitsConfig(max_page_pixels=1_000_000)
    with pytest.raises(InputRejected) as big:
        image_to_pdf(_encode(_page(size=(1200, 1000)), "PNG"), small)
    assert big.value.code in {ReasonCode.INPUT_TOO_LARGE, ReasonCode.INPUT_UNREADABLE}
    with pytest.raises(InputRejected) as bad:
        image_to_pdf(b"II*\x00" + b"\x00" * 64, LimitsConfig())
    assert bad.value.code is ReasonCode.INPUT_UNREADABLE
