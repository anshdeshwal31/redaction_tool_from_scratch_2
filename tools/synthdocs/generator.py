"""Generate synthetic IME-style letters with planted (fictitious) PII.

Two renditions of the same content:

* ``digital`` — born-digital PDF with a real text layer (base-14 Helvetica), and
* ``scanned`` — the page rendered to a 300 dpi image (optionally degraded: skew, blur, noise,
  JPEG artefacts) embedded as an image-only PDF, i.e. what a scanner produces.

``Planted.phase`` marks from which development phase the detector suite is expected to find the
value; tests assert only what the current phase supports, and tighten as phases land.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pikepdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from mlredact.resources.loader import font_path

PAGE_W_PT, PAGE_H_PT = 595.276, 841.89  # A4


@dataclass(frozen=True)
class Planted:
    kind: str  # EntityType value
    text: str = field(repr=False)
    phase: int = 1  # first phase expected to remove it


@dataclass(frozen=True)
class SynthDoc:
    pdf: bytes
    planted: tuple[Planted, ...]
    lines: tuple[str, ...]
    # Ground truth for non-text artefacts: (region kind, (x0, y0, x1, y1) in pixels at the scan dpi).
    artefacts: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()


def medicare_number(stem8: str, issue: int = 1) -> str:
    weights = (1, 3, 7, 9, 1, 3, 7, 9)
    check = sum(int(d) * w for d, w in zip(stem8, weights, strict=True)) % 10
    raw = f"{stem8}{check}{issue}"
    return f"{raw[:4]} {raw[4:9]} {raw[9]}"


def provider_number(stem6: str, location: str = "A") -> str:
    plv = {**{str(i): i for i in range(10)}, **dict(zip("ABCDEFGHJKLMNPQRTUVWXY", range(10, 32), strict=True))}
    weights = (3, 5, 8, 4, 2, 1)
    total = sum(int(d) * w for d, w in zip(stem6, weights, strict=True)) + plv[location] * 6
    return f"{stem6}{location}{'YXWTLKJHFBA'[total % 11]}"


def _content() -> tuple[list[str], list[Planted]]:
    medicare = medicare_number("29537015", 1)
    provider = provider_number("212345", "A")
    lines = [
        "HARBOURSIDE MEDICOLEGAL SERVICES",
        "Level 4, 123 George Street, Sydney NSW 2000",
        "Ph: (02) 9876 5432   Email: admin@harbourside-medicolegal.example",
        "",
        "12 June 2024",
        "",
        "Ms Jane Citizen",
        "Smith & Partners Lawyers",
        "PO Box 123, Parramatta NSW 2150",
        "",
        "Our Ref: HMS-2024-0187        Your Ref: SP/4471/JC",
        "",
        "Re: Mr John Andrew SMITH      DOB: 14/03/1978",
        f"Claim No: WC1234567      Medicare No: {medicare}",
        "Address: 42 Wattle Grove Road, Penrith NSW 2750",
        "",
        "Dear Ms Citizen,",
        "",
        "I examined Mr Smith on 2 May 2024 at my rooms in relation to a workplace",
        "injury sustained on 3 February 2023. He reports constant low back pain",
        "radiating to the left leg. Tinel's sign was negative and Phalen's test was",
        "negative bilaterally. He takes Panadeine Forte as required.",
        "",
        "Mr Smith can be contacted on mobile 0412 345 678 or by email at",
        "john.smith78@example.org regarding further appointments.",
        "",
        "Yours sincerely,",
        "",
        "Dr Peter Brown FRACS",
        f"Provider No: {provider}",
    ]
    planted = [
        Planted("street_address", "123 George Street", 1),
        Planted("phone", "(02) 9876 5432", 1),
        Planted("email", "admin@harbourside-medicolegal.example", 1),
        Planted("person", "Jane Citizen", 1),
        Planted("reference", "HMS-2024-0187", 1),
        Planted("reference", "SP/4471/JC", 1),
        Planted("person", "John Andrew SMITH", 1),
        Planted("date_of_birth", "14/03/1978", 1),
        Planted("reference", "WC1234567", 1),
        Planted("medicare", medicare, 1),
        Planted("street_address", "42 Wattle Grove Road", 1),
        Planted("locality", "Penrith", 1),
        Planted("postcode", "2750", 1),
        Planted("phone", "0412 345 678", 1),
        Planted("email", "john.smith78@example.org", 1),
        Planted("person", "Peter Brown", 1),
        Planted("provider_number", provider, 1),
        Planted("person", "Smith", 1),  # "Mr Smith": titles are kept, the name is removed
        Planted("organisation", "Smith & Partners Lawyers", 3),
        Planted("locality", "Parramatta", 1),
    ]
    return lines, planted


def _escape_pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _digital_pdf(lines: list[str]) -> bytes:
    pdf = pikepdf.new()
    font = pikepdf.Dictionary(
        Type=pikepdf.Name.Font,
        Subtype=pikepdf.Name.Type1,
        BaseFont=pikepdf.Name.Helvetica,
        Encoding=pikepdf.Name.WinAnsiEncoding,
    )
    ops = ["BT", "/F1 11 Tf", "13 TL", f"60 {PAGE_H_PT - 60:.2f} Td"]
    for line in lines:
        ops.append(f"({_escape_pdf_text(line)}) Tj T*")
    ops.append("ET")
    content = pikepdf.Stream(pdf, "\n".join(ops).encode("latin-1"))
    page = pikepdf.Dictionary(
        Type=pikepdf.Name.Page,
        MediaBox=[0, 0, PAGE_W_PT, PAGE_H_PT],
        Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)),
        Contents=content,
    )
    pdf.pages.append(pikepdf.Page(page))
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()


def _scan_image(lines: list[str], dpi: int, seed: int, degrade: bool) -> Image.Image:
    page = _typeset_page(lines, dpi)
    return _degrade(page, seed) if degrade else page


def _typeset_page(lines: list[str], dpi: int) -> Image.Image:
    scale = dpi / 72.0
    w, h = round(PAGE_W_PT * scale), round(PAGE_H_PT * scale)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path("Sans", "Regular")), size=round(11 * scale))
    y = 60 * scale
    for line in lines:
        draw.text((60 * scale, y), line, fill=20, font=font)
        y += 13 * scale
    return img


def _degrade(img: Image.Image, seed: int) -> Image.Image:
    """Skew, blur, sensor noise and JPEG artefacts (grey or colour scans)."""
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-0.8, 0.8))
    fill: int | tuple[int, int, int] = 255 if img.mode == "L" else (255, 255, 255)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, 9.0, size=arr.shape[:2])
    arr = arr + (noise if arr.ndim == 2 else noise[..., None])
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=img.mode)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=55)
    return Image.open(io.BytesIO(buf.getvalue())).convert(img.mode)


Box = tuple[int, int, int, int]


def _scribble(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    size: tuple[float, float],
    rng: np.random.Generator,
    fill: tuple[int, int, int],
    width: int,
) -> Box:
    """Cursive-like pen strokes, unreadable on purpose (looping sums of sines)."""
    (x0, y0), (w, h) = origin, size
    t = np.linspace(0.0, 1.0, 400)
    k1, k2, k3 = rng.uniform(5, 9), rng.uniform(11, 17), rng.uniform(0, 6.28)
    xs = x0 + w * t + 0.04 * w * np.sin(2 * np.pi * k2 * t + k3)
    ys = y0 + h / 2 + 0.38 * h * np.sin(2 * np.pi * k1 * t) + 0.12 * h * np.sin(2 * np.pi * k2 * t)
    draw.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=fill, width=width, joint="curve")
    return (int(xs.min()) - width, int(ys.min()) - width, int(xs.max()) + width + 1, int(ys.max()) + width + 1)


def _add_artefacts(
    page: Image.Image, lines: list[str], dpi: int, seed: int
) -> tuple[Image.Image, list[tuple[str, Box]]]:
    """Signature, QR code, red stamp, margin handwriting and show-through on a colour scan."""
    import zxingcpp

    rng = np.random.default_rng(seed + 1000)
    scale = dpi / 72.0
    img = page.convert("RGB")
    draw = ImageDraw.Draw(img)
    truth: list[tuple[str, Box]] = []

    def line_y(text: str) -> float:
        return (60 + 13 * lines.index(text)) * scale

    # Pen signature between "Yours sincerely," and the typed name (black ink, like most scans).
    y_sig = line_y("Yours sincerely,") + 13 * scale
    truth.append(
        ("signature", _scribble(draw, (70 * scale, y_sig - 2 * scale), (150 * scale, 18 * scale), rng, (25, 25, 35), 5))
    )

    # QR code carrying the claimant's details, top-right corner.
    barcode = zxingcpp.create_barcode("SMITH John Andrew DOB 14/03/1978 Claim WC1234567", zxingcpp.BarcodeFormat.QRCode)
    qr = Image.fromarray(
        np.asarray(zxingcpp.write_barcode_to_image(barcode, scale=max(1, round(1.4 * scale))), dtype=np.uint8)
    )
    qr = qr.convert("RGB")
    qx, qy = round(470 * scale), round(28 * scale)
    img.paste(qr, (qx, qy))
    truth.append(("barcode", (qx, qy, qx + qr.width, qy + qr.height)))

    # Red "RECEIVED" date stamp below the letter.
    stamp = Image.new("RGBA", (round(170 * scale), round(55 * scale)), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    red = (200, 30, 40, 255)
    sd.rectangle([2, 2, stamp.width - 3, stamp.height - 3], outline=red, width=round(2 * scale))
    sfont = ImageFont.truetype(str(font_path("Sans", "Bold")), size=round(14 * scale))
    sd.text((round(12 * scale), round(6 * scale)), "RECEIVED", fill=red, font=sfont)
    sd.text((round(12 * scale), round(28 * scale)), "17 JUN 2024", fill=red, font=sfont)
    stamp = stamp.rotate(-7, expand=True, resample=Image.Resampling.BICUBIC)
    sx, sy = round(330 * scale), round(530 * scale)
    img.paste(stamp, (sx, sy), stamp)
    truth.append(("stamp", (sx, sy, sx + stamp.width, sy + stamp.height)))

    # Unreadable handwriting in the left margin, next to the "Re:" line.
    y_re = line_y(next(t for t in lines if t.startswith("Re:")))
    truth.append(
        ("handwriting", _scribble(draw, (8 * scale, y_re - 30 * scale), (34 * scale, 60 * scale), rng, (30, 30, 30), 4))
    )

    # Show-through: mirrored text from the reverse of the sheet, very faint.
    ghost = Image.new("L", (round(330 * scale), round(40 * scale)), 0)
    gd = ImageDraw.Draw(ghost)
    gfont = ImageFont.truetype(str(font_path("Sans", "Regular")), size=round(11 * scale))
    gd.text((0, 0), "Patient: Mary SMITH  DOB 02/11/1980", fill=255, font=gfont)
    gd.text((0, round(14 * scale)), "Mobile 0499 888 777  Medicare 4421 55667 1", fill=255, font=gfont)
    ghost = ImageOps.mirror(ghost)
    gx, gy = round(60 * scale), round(640 * scale)
    img.paste(Image.new("RGB", ghost.size, (222, 222, 222)), (gx, gy), ghost)
    gb = ghost.getbbox() or (0, 0, ghost.width, ghost.height)
    truth.append(("faint_ink", (gx + gb[0], gy + gb[1], gx + gb[2], gy + gb[3])))
    return img, truth


def _image_pdf(img: Image.Image) -> bytes:
    pdf = pikepdf.new()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    image = pikepdf.Stream(pdf, buf.getvalue())
    image.Type = pikepdf.Name.XObject
    image.Subtype = pikepdf.Name.Image
    image.Width, image.Height = img.size
    image.ColorSpace = pikepdf.Name.DeviceGray if img.mode == "L" else pikepdf.Name.DeviceRGB
    image.BitsPerComponent = 8
    image.Filter = pikepdf.Name.DCTDecode
    content = pikepdf.Stream(pdf, f"q {PAGE_W_PT:.3f} 0 0 {PAGE_H_PT:.3f} 0 0 cm /Im0 Do Q".encode())
    page = pikepdf.Dictionary(
        Type=pikepdf.Name.Page,
        MediaBox=[0, 0, PAGE_W_PT, PAGE_H_PT],
        Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=image)),
        Contents=content,
    )
    pdf.pages.append(pikepdf.Page(page))
    out = io.BytesIO()
    pdf.save(out, deterministic_id=True)
    return out.getvalue()


def make_ime_letter(
    *, scanned: bool, degrade: bool = False, seed: int = 0, dpi: int = 300, artefacts: bool = False
) -> SynthDoc:
    """``artefacts`` (scanned only): add a signature, QR code, red stamp, margin handwriting and
    show-through; their pixel boxes are returned in ``SynthDoc.artefacts``."""
    lines, planted = _content()
    if not scanned:
        return SynthDoc(pdf=_digital_pdf(lines), planted=tuple(planted), lines=tuple(lines))
    page = _typeset_page(lines, dpi)
    truth: list[tuple[str, Box]] = []
    if artefacts:
        page, truth = _add_artefacts(page, lines, dpi, seed)
    if degrade:
        page = _degrade(page, seed)
    return SynthDoc(pdf=_image_pdf(page), planted=tuple(planted), lines=tuple(lines), artefacts=tuple(truth))


def combine(pdfs: list[bytes]) -> bytes:
    """Concatenate single-document PDFs into one multi-page PDF (mixed scanned/digital pages)."""
    out = pikepdf.new()
    for data in pdfs:
        with pikepdf.open(io.BytesIO(data)) as src:
            out.pages.extend(src.pages)
    buf = io.BytesIO()
    out.save(buf, deterministic_id=True)
    return buf.getvalue()


def encrypted(data: bytes, user_password: str = "secret") -> bytes:  # noqa: S107 - test fixture
    with pikepdf.open(io.BytesIO(data)) as pdf:
        buf = io.BytesIO()
        pdf.save(buf, encryption=pikepdf.Encryption(user=user_password, owner=user_password + "-owner"))
        return buf.getvalue()
