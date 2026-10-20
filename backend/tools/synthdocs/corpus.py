"""Nine synthetic scanned Australian medico-legal documents in three difficulty tiers, with answer keys.

Each document (``corpus_docs``) is written in a light markup, typeset onto page images, "scanned"
(skew, blur, noise, stains, fax, ...) and saved as an image-only PDF next to an answer key that
lists every planted value and what the client profile does with it.  All content is fictitious;
phone numbers come from the ACMA ranges reserved for fiction.

    PYTHONPATH=tools .venv/Scripts/python -m synthdocs.corpus --out out/synth_corpus   (from backend/)

Markup inside any text:
    [value](kind)    a planted value; ``kind`` is an EntityType value (person, medicare, date, ...)
    <value>(class)   content that must survive (medication, eponym, homograph, clinical, citation, ...)

Block lines (anything else is a wrapped paragraph; "" is a paragraph gap):
    @h1 / @h2 / @h3 / @bold / @center / @right / @small text
    @li label | text          hanging-indent list item
    @rule                     horizontal line
    @gap pt                   vertical space
    @font Sans|Serif|Mono     @size pt
    @pre text                 monospaced line, spacing kept
    @table w1,w2,...          rows "a | b | c" until @end; the first row is the header
    @field Label | value ;; Label | value      boxed form fields; "~value" is handwritten
    @write lines | ~text      ruled box filled in by hand
    @check x | text           ticked checkbox ("-" for an empty one)
    @sig                      pen signature at the current position
Overlays at absolute positions in points (later ones lie on top):
    @stamp x,y,angle | line | line          @hand x,y,angle | line | line
    @qr x,y,size | payload                  @label x,y,barcode | line | line   (patient sticker)
    @logo x,y | AB                          @photo x,y,w,h
    @box x,y,w,h                            @text x,y,size,style | text
    @sig x,y
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pikepdf
import yaml
import zxingcpp
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from mlredact.config import profiles as _profiles
from mlredact.resources.loader import font_path
from synthdocs.generator import _scribble

PORTRAIT = (595.276, 841.89)  # A4
LANDSCAPE = (PORTRAIT[1], PORTRAIT[0])
MARGIN = 56.0
TOP = 64.0
INK = (25, 25, 28)
BLUE = (28, 48, 150)
RED = (200, 30, 40)
TEAL = (0, 105, 115)

_MARKUP = re.compile(r"\[([^\]]+)\]\(([a-z_]+)\)|<([^>]+)>\(([a-z_]+)\)")
PROTECTED = {"medication", "eponym", "homograph", "clinical", "citation", "boilerplate", "role"}


@dataclass(frozen=True)
class Scan:
    dpi: int = 300
    colour: bool = False
    skew: float = 0.3  # largest page tilt, degrees
    blur: float = 0.4  # Gaussian radius at 300 dpi
    noise: float = 4.0
    jpeg: int = 80
    paper: tuple[int, int, int] = (255, 255, 255)
    fade: float = 0.0  # 0..1: lifts the ink towards the paper (worn toner, light photocopy)
    shadow: float = 0.0  # 0..1: darkening at the binding edge
    speckle: int = 0  # dust specks per page
    show_through: float = 0.0  # opacity of the mirrored back of the sheet
    fax: str | None = None  # fax header template ({page}, {pages}); also makes the pages bilevel
    copier: bool = False  # dark copier edges and punch holes
    coffee: tuple[int, ...] = ()  # 0-based pages with a coffee ring


@dataclass
class Page:
    lines: list[str]
    landscape: bool = False  # typeset landscape, scanned a quarter turn into a portrait page
    upside_down: bool = False


@dataclass
class DocSpec:
    slug: str
    tier: str
    title: str
    challenges: list[str]
    pages: list[Page]
    scan: Scan
    font: str = "Sans"
    size: float = 10.5
    header: str | None = None  # running header from the second page on ({page}, {pages})
    footer: str | None = None  # running footer on every page
    seed: int = 0


@dataclass
class Item:
    page: int  # 1-based
    kind: str
    text: str
    action: str  # what the client profile does with it (surrogate, keep, ...)
    where: str = "body"


@dataclass
class Region:
    page: int
    kind: str
    action: str
    box_pt: tuple[float, float, float, float]  # on the typeset page, before scanning
    note: str = ""


class _Key:
    """The answer key, filled in while the markup is stripped."""

    def __init__(self) -> None:
        policy = yaml.safe_load((Path(_profiles.__file__).parent / "client.yaml").read_text("utf-8"))["policy"]
        self.entity_actions: dict[str, str] = policy["entity_actions"]
        self.region_actions: dict[str, str] = policy["region_actions"]
        self.items: list[Item] = []
        self.regions: list[Region] = []

    def strip(self, text: str, page: int, where: str) -> str:
        def sub(m: re.Match[str]) -> str:
            if m.group(1) is not None:
                value, kind = m.group(1), m.group(2)
                if kind not in self.entity_actions:
                    raise ValueError(f"unknown entity kind {kind!r} in {text!r}")
                self.items.append(Item(page + 1, kind, value, self.entity_actions[kind], where))
            else:
                value, kind = m.group(3), m.group(4)
                if kind not in PROTECTED:
                    raise ValueError(f"unknown protected class {kind!r} in {text!r}")
                self.items.append(Item(page + 1, kind, value, "keep", where))
            return value

        return _MARKUP.sub(sub, text)

    def region(self, page: int, kind: str, box: tuple[float, float, float, float], note: str = "") -> None:
        self.regions.append(
            Region(page + 1, kind, self.region_actions[kind], tuple(round(v, 1) for v in box), note)  # type: ignore[arg-type]
        )


def unmark(text: str) -> str:
    return _MARKUP.sub(lambda m: m.group(1) if m.group(1) is not None else m.group(3), text)


_FONTS: dict[tuple[str, str, int], ImageFont.FreeTypeFont] = {}


def _font(family: str, style: str, px: float) -> ImageFont.FreeTypeFont:
    if family == "Mono":
        style = "Regular"
    key = (family, style, max(4, round(px)))
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(str(font_path(family, style)), key[2])
    return _FONTS[key]


def _hand(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, px: float, rng: np.random.Generator, fill: tuple
) -> tuple[float, float]:
    """Neat pen lettering: italic serif with a wandering baseline and uneven spacing. Returns the end point."""
    f = _font("Serif", "Italic", px)
    x, y = xy
    stroke = max(0, round(px / 40))
    drift = 0.0
    for ch in text:
        drift = 0.7 * drift + rng.normal(0, 0.035 * px)
        draw.text((x, y + drift), ch, font=f, fill=fill, stroke_width=stroke, stroke_fill=fill)
        x += f.getlength(ch) * rng.uniform(0.97, 1.09)
    return x, y + px


class _Typesetter:
    def __init__(self, spec: DocSpec, page: Page, index: int, key: _Key, rng: np.random.Generator) -> None:
        self.spec, self.index, self.key, self.rng = spec, index, key, rng
        self.w, self.h = LANDSCAPE if page.landscape else PORTRAIT
        self.s = spec.scan.dpi / 72.0
        self.img = Image.new("RGB", (round(self.w * self.s), round(self.h * self.s)), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.img)
        self.family, self.size = spec.font, spec.size
        self.y = TOP

    # -- primitives (points in, pixels out) ----------------------------------------------------

    def px(self, v: float) -> float:
        return v * self.s

    def width(self, text: str, size: float, style: str = "Regular", family: str | None = None) -> float:
        return _font(family or self.family, style, self.px(size)).getlength(text) / self.s

    def put(
        self,
        x: float,
        y: float,
        text: str,
        size: float,
        style: str = "Regular",
        family: str | None = None,
        fill: tuple = INK,
    ) -> None:
        self.draw.text(
            (self.px(x), self.px(y)), text, font=_font(family or self.family, style, self.px(size)), fill=fill
        )

    def rect(self, x0: float, y0: float, x1: float, y1: float, width: float = 0.8, radius: float = 0) -> None:
        box = [self.px(x0), self.px(y0), self.px(x1), self.px(y1)]
        self.draw.rounded_rectangle(box, radius=self.px(radius), outline=INK, width=max(1, round(self.px(width))))

    def strip(self, text: str, where: str = "body") -> str:
        return self.key.strip(text, self.index, where)

    def room(self, need: float) -> None:
        if self.y + need > self.h - 46:
            raise ValueError(f"{self.spec.slug}: page {self.index + 1} overflows")

    def wrap(
        self, text: str, width: float, size: float, style: str = "Regular", family: str | None = None
    ) -> list[str]:
        lines, cur = [], ""
        for word in text.split(" "):
            cand = f"{cur} {word}" if cur else word
            if cur and self.width(cand, size, style, family) > width:
                lines.append(cur)
                cur = word
            else:
                cur = cand
        return [*lines, cur] if cur else lines or [""]

    def paragraph(
        self,
        raw: str,
        size: float | None = None,
        style: str = "Regular",
        align: str = "left",
        x0: float = MARGIN,
        x1: float | None = None,
    ) -> None:
        size = size or self.size
        x1 = self.w - MARGIN if x1 is None else x1
        lead = size * 1.32
        for line in self.wrap(self.strip(raw), x1 - x0, size, style):
            lw = self.width(line, size, style)
            x = {"left": x0, "center": (x0 + x1 - lw) / 2, "right": x1 - lw}[align]
            self.room(lead)
            self.put(x, self.y, line, size, style)
            self.y += lead

    # -- blocks ----------------------------------------------------------------------------------

    def block(self, line: str, rest_of_page: Iterator[str]) -> None:
        if not line:
            self.y += self.size * 0.6
            return
        if not line.startswith("@"):
            self.paragraph(line)
            return
        cmd, _, arg = line.partition(" ")
        head, _, body = arg.partition(" | ")
        nums = [float(v) for v in head.split(",")] if re.fullmatch(r"-?[\d.]+(,-?[\d.]+)*", head) else []
        if cmd == "@h1":
            self.y += 4
            self.paragraph(arg, self.size + 4.5, "Bold", "center")
            self.y += 4
        elif cmd == "@h2":
            self.y += 6
            self.paragraph(arg, self.size + 1.5, "Bold")
            self.y += 2
        elif cmd == "@h3":
            self.y += 3
            self.paragraph(arg, self.size, "Bold")
        elif cmd == "@bold":
            self.paragraph(arg, style="Bold")
        elif cmd in ("@center", "@right"):
            self.paragraph(arg, align=cmd[1:])
        elif cmd == "@small":
            self.paragraph(arg, self.size - 2)
        elif cmd == "@li":
            label, _, text = arg.partition(" | ")
            self.room(self.size * 1.32)
            self.put(MARGIN + 6, self.y, label, self.size)
            self.paragraph(text, x0=MARGIN + 26)
        elif cmd == "@rule":
            self.y += 3
            self.draw.line(
                [self.px(MARGIN), self.px(self.y), self.px(self.w - MARGIN), self.px(self.y)],
                fill=INK,
                width=max(1, round(self.px(0.8))),
            )
            self.y += 7
        elif cmd == "@gap":
            self.y += float(arg)
        elif cmd == "@font":
            self.family = arg
        elif cmd == "@size":
            self.size = float(arg)
        elif cmd == "@pre":
            self.room(self.size * 1.25)
            self.put(MARGIN, self.y, self.strip(arg), self.size, family="Mono")
            self.y += self.size * 1.25
        elif cmd == "@table":
            rows = []
            for row in rest_of_page:
                if row == "@end":
                    break
                rows.append(row)
            self.table([float(v) for v in arg.split(",")], rows)
        elif cmd == "@field":
            self.fields(arg)
        elif cmd == "@write":
            self.write_box(int(head), body)
        elif cmd == "@check":
            self.check(head.strip() == "x", body)
        elif cmd == "@sig":
            x, y = (nums[0], nums[1]) if nums else (MARGIN + 8, self.y)
            self.signature(x, y)
            if not nums:
                self.y += 30
        elif cmd == "@stamp":
            self.stamp(nums[0], nums[1], nums[2], body.split(" | "))
        elif cmd == "@hand":
            self.margin_note(nums[0], nums[1], nums[2], body.split(" | "))
        elif cmd == "@qr":
            self.qr(nums[0], nums[1], nums[2], unmark(body))
        elif cmd == "@label":
            x, y, barcode = head.split(",", 2)
            self.sticker(float(x), float(y), barcode, body.split(" | "))
        elif cmd == "@logo":
            self.logo(nums[0], nums[1], body)
        elif cmd == "@photo":
            self.photo(*nums)
        elif cmd == "@box":
            self.rect(nums[0], nums[1], nums[0] + nums[2], nums[1] + nums[3], 1.0, radius=8)
        elif cmd == "@text":
            parts = head.split(",")
            self.put(float(parts[0]), float(parts[1]), self.strip(body), float(parts[2]), parts[3])
        else:
            raise ValueError(f"{self.spec.slug}: unknown block {cmd}")

    def table(self, widths: list[float], rows: list[str]) -> None:
        size, pad = self.size - 0.5, 3.0
        lead = size * 1.28
        xs = [MARGIN + sum(widths[:i]) for i in range(len(widths))]
        for r, row in enumerate(rows):
            style = "Bold" if r == 0 else "Regular"
            cells = [c.strip() for c in row.split("|")]
            if len(cells) != len(widths):
                raise ValueError(f"{self.spec.slug}: table row has {len(cells)} cells: {row!r}")
            wrapped = [
                self.wrap(self.strip(c, "table"), w - 2 * pad, size, style) for c, w in zip(cells, widths, strict=True)
            ]
            height = max(len(w) for w in wrapped) * lead + 2 * pad
            self.room(height)
            for x, w, lines in zip(xs, widths, wrapped, strict=True):
                self.rect(x, self.y, x + w, self.y + height, 0.7)
                for k, text in enumerate(lines):
                    self.put(x + pad, self.y + pad + k * lead, text, size, style)
            self.y += height
        self.y += 6

    def fields(self, arg: str) -> None:
        parts = [p.split("|", 1) for p in arg.split(";;")]
        gap, height = 6.0, 32.0
        bw = (self.w - 2 * MARGIN - gap * (len(parts) - 1)) / len(parts)
        self.room(height + 4)
        for i, (label, value) in enumerate(parts):
            x = MARGIN + i * (bw + gap)
            self.rect(x, self.y, x + bw, self.y + height, 0.7)
            self.put(x + 3, self.y + 2, label.strip(), 7.0, fill=(70, 70, 70))
            value = value.strip()
            hand = value.startswith("~")
            text = self.strip(value.lstrip("~"), "handwriting" if hand else "form")
            size = 12.5 if hand else 10.0
            while (
                size > 7 and self.width(text, size, "Italic" if hand else "Regular", "Serif" if hand else None) > bw - 8
            ):
                size -= 0.5
            if hand:
                _hand(self.draw, (self.px(x + 5), self.px(self.y + 13)), text, self.px(size), self.rng, BLUE)
            else:
                self.put(x + 4, self.y + 15, text, size)
        self.y += height + 4

    def write_box(self, n_lines: int, raw: str) -> None:
        """A ruled area filled in by hand."""
        lead, size = 19.0, 12.0
        height = n_lines * lead + 6
        self.room(height + 4)
        x0, x1 = MARGIN, self.w - MARGIN
        self.rect(x0, self.y, x1, self.y + height, 0.7)
        for k in range(1, n_lines + 1):
            yy = self.px(self.y + 3 + k * lead)
            self.draw.line([self.px(x0 + 4), yy, self.px(x1 - 4), yy], fill=(150, 150, 150), width=1)
        text = self.strip(raw.lstrip("~"), "handwriting")
        lines = self.wrap(text, x1 - x0 - 16, size, "Italic", "Serif")
        if len(lines) > n_lines:
            raise ValueError(f"{self.spec.slug}: handwriting does not fit in {n_lines} lines")
        for k, line in enumerate(lines):
            _hand(self.draw, (self.px(x0 + 8), self.px(self.y + 3 + k * lead + 3)), line, self.px(size), self.rng, BLUE)
        self.y += height + 4

    def check(self, ticked: bool, raw: str) -> None:
        self.room(16)
        x, y, b = MARGIN + 4, self.y + 1, 9.0
        self.rect(x, y, x + b, y + b, 0.8)
        if ticked:
            w = max(2, round(self.px(1.4)))
            self.draw.line([self.px(x - 1), self.px(y - 1), self.px(x + b + 1), self.px(y + b + 1)], fill=BLUE, width=w)
            self.draw.line([self.px(x + b + 1), self.px(y - 1), self.px(x - 1), self.px(y + b + 1)], fill=BLUE, width=w)
        self.paragraph(raw, x0=MARGIN + 22)
        self.y += 2

    # -- non-text regions --------------------------------------------------------------------------

    def signature(self, x: float, y: float) -> None:
        box = _scribble(
            self.draw,
            (self.px(x), self.px(y)),
            (self.px(120), self.px(24)),
            self.rng,
            (20, 25, 60),
            max(2, round(self.px(1.1))),
        )
        self.key.region(self.index, "signature", tuple(v / self.s for v in box))  # type: ignore[arg-type]

    def _paste_rotated(self, layer: Image.Image, x: float, y: float, angle: float) -> tuple[float, ...]:
        layer = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        px, py = round(self.px(x)), round(self.px(y))
        self.img.paste(layer, (px, py), layer)
        return (x, y, x + layer.width / self.s, y + layer.height / self.s)

    def stamp(self, x: float, y: float, angle: float, lines: list[str]) -> None:
        texts = [self.strip(t, "stamp") for t in lines]
        size = 11.0
        f = _font("Sans", "Bold", self.px(size))
        w = max(f.getlength(t) for t in texts) + self.px(20)
        h = self.px(size * 1.35 * len(texts) + 14)
        layer = Image.new("RGBA", (round(w), round(h)), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        ink = (*RED, 215)
        d.rounded_rectangle([2, 2, w - 3, h - 3], radius=self.px(4), outline=ink, width=max(2, round(self.px(1.6))))
        for k, t in enumerate(texts):
            tw = f.getlength(t)
            d.text(((w - tw) / 2, self.px(7 + k * size * 1.35)), t, font=f, fill=ink)
        self.key.region(self.index, "stamp", self._paste_rotated(layer, x, y, angle))  # type: ignore[arg-type]

    def margin_note(self, x: float, y: float, angle: float, lines: list[str]) -> None:
        texts = [self.strip(t, "handwriting") for t in lines]
        px = self.px(12.5)
        f = _font("Serif", "Italic", px)
        w = max(f.getlength(t) for t in texts) * 1.12 + px
        h = px * 1.45 * len(texts) + px * 0.6
        layer = Image.new("RGBA", (round(w), round(h)), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for k, t in enumerate(texts):
            _hand(d, (px * 0.3, px * 0.2 + k * px * 1.45), t, px, self.rng, (*BLUE, 255))
        self.key.region(self.index, "handwriting", self._paste_rotated(layer, x, y, angle))  # type: ignore[arg-type]

    def qr(self, x: float, y: float, size: float, payload: str) -> None:
        code = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
        img = Image.fromarray(np.asarray(zxingcpp.write_barcode_to_image(code, scale=4), dtype=np.uint8))
        img = img.convert("RGB").resize((round(self.px(size)), round(self.px(size))), Image.Resampling.NEAREST)
        self.img.paste(img, (round(self.px(x)), round(self.px(y))))
        self.key.region(self.index, "barcode", (x, y, x + size, y + size), f"QR code encoding: {payload}")

    def sticker(self, x: float, y: float, barcode: str, lines: list[str]) -> None:
        """Hospital patient label: a few lines of details over a Code 128 barcode."""
        size, lead, w = 8.0, 10.5, 200.0
        h = len(lines) * lead + 34
        self.rect(x, y, x + w, y + h, 0.9, radius=5)
        for k, raw in enumerate(lines):
            self.put(x + 6, y + 5 + k * lead, self.strip(raw, "label"), size, "Bold" if k == 0 else "Regular")
        code = zxingcpp.create_barcode(unmark(barcode), zxingcpp.BarcodeFormat.Code128)
        bars = np.asarray(zxingcpp.write_barcode_to_image(code, scale=2), dtype=np.uint8)
        if bars.ndim == 2 and bars.shape[0] == 1:
            bars = np.repeat(bars, 10, axis=0)
        bx, by, bw, bh = x + 6, y + 8 + len(lines) * lead, 150.0, 20.0
        img = (
            Image.fromarray(bars)
            .convert("RGB")
            .resize((round(self.px(bw)), round(self.px(bh))), Image.Resampling.NEAREST)
        )
        self.img.paste(img, (round(self.px(bx)), round(self.px(by))))
        self.key.region(self.index, "barcode", (bx, by, bx + bw, by + bh), f"barcode encoding: {unmark(barcode)}")

    def logo(self, x: float, y: float, initials: str) -> None:
        d = 40.0
        box = [self.px(x), self.px(y), self.px(x + d), self.px(y + d)]
        self.draw.ellipse(box, fill=TEAL)
        inner = [self.px(x + 4), self.px(y + 4), self.px(x + d - 4), self.px(y + d - 4)]
        self.draw.ellipse(inner, outline=(255, 255, 255), width=max(1, round(self.px(1.2))))
        f = _font("Serif", "Bold", self.px(15))
        tw = f.getlength(initials)
        self.draw.text((self.px(x + d / 2) - tw / 2, self.px(y + 11)), initials, font=f, fill=(255, 255, 255))
        self.key.region(self.index, "logo", (x, y, x + d, y + d))

    def photo(self, x: float, y: float, w: float, h: float) -> None:
        self.draw.rectangle([self.px(x), self.px(y), self.px(x + w), self.px(y + h)], fill=(205, 205, 200))
        cx = x + w / 2
        self.draw.ellipse(
            [self.px(cx - w * 0.22), self.px(y + h * 0.14), self.px(cx + w * 0.22), self.px(y + h * 0.56)],
            fill=(110, 100, 95),
        )
        self.draw.ellipse(
            [self.px(x + w * 0.08), self.px(y + h * 0.62), self.px(x + w * 0.92), self.px(y + h * 1.25)],
            fill=(80, 85, 100),
        )
        self.draw.rectangle([self.px(x), self.px(y + h), self.px(x + w), self.px(y + h * 1.3)], fill=(255, 255, 255))
        self.rect(x, y, x + w, y + h, 0.6)
        self.key.region(self.index, "photo", (x, y, x + w, y + h))

    def running(self, template: str, y: float, where: str, n_pages: int) -> None:
        text = self.strip(template.replace("{page}", str(self.index + 1)).replace("{pages}", str(n_pages)), where)
        size = 8.0
        if where == "header":
            self.put(MARGIN, y, text, size, "Bold")
            self.draw.line(
                [self.px(MARGIN), self.px(y + 12), self.px(self.w - MARGIN), self.px(y + 12)],
                fill=INK,
                width=max(1, round(self.px(0.5))),
            )
        else:
            self.put((self.w - self.width(text, size)) / 2, y, text, size, fill=(60, 60, 60))


# -- scanning ----------------------------------------------------------------------------------------


def _typeset(spec: DocSpec, key: _Key) -> list[Image.Image]:
    rng = np.random.default_rng(spec.seed + 99)
    images = []
    for i, page in enumerate(spec.pages):
        t = _Typesetter(spec, page, i, key, rng)
        lines = iter(page.lines)
        for line in lines:
            t.block(line, lines)
        if spec.header and i > 0:
            t.running(spec.header, 28, "header", len(spec.pages))
        if spec.footer:
            t.running(spec.footer, t.h - 32, "footer", len(spec.pages))
        images.append(t.img)
    return images


def _fax_header(img: Image.Image, template: str, page: int, n_pages: int, key: _Key, dpi: int) -> Image.Image:
    text = key.strip(
        template.replace("{page}", f"{page + 1:03d}").replace("{pages}", f"{n_pages:03d}"), page, "fax header"
    )
    s = dpi / 72.0
    img = img.copy()
    ImageDraw.Draw(img).text((18 * s, 8 * s), text, font=_font("Mono", "Regular", 7.5 * s), fill=INK)
    return img


def _blend(img: Image.Image, arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=img.mode)


def _scan_page(
    img: Image.Image, back: Image.Image | None, spec: DocSpec, i: int, key: _Key, rng: np.random.Generator
) -> Image.Image:
    sc, page, n = spec.scan, spec.pages[i], len(spec.pages)
    s = sc.dpi / 72.0
    if back is not None and sc.show_through:
        ghost = back.transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize(img.size).convert("L")
        a = np.asarray(img, np.float32)
        g = (255.0 - np.asarray(ghost, np.float32)) * sc.show_through
        img = _blend(img, np.minimum(a, (255.0 - g)[..., None]))
        w_pt, h_pt = LANDSCAPE if page.landscape else PORTRAIT
        key.region(
            i, "faint_ink", (0, 0, w_pt, h_pt), "mirrored text from the back of the sheet (another page's names)"
        )
    if page.landscape:
        img = img.transpose(Image.Transpose.ROTATE_90)
    if page.upside_down:
        img = img.transpose(Image.Transpose.ROTATE_180)
    if sc.fax:
        img = _fax_header(img, sc.fax, i, n, key, sc.dpi)
    a = np.asarray(img, np.float32)
    if sc.fade:
        a = 255.0 - (255.0 - a) * (1.0 - sc.fade)
    a = a * (np.asarray(sc.paper, np.float32) / 255.0)
    img = _blend(img, a)
    img = img.rotate(float(rng.uniform(-sc.skew, sc.skew)), resample=Image.Resampling.BICUBIC, fillcolor=sc.paper)
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if sc.shadow:
        a = a * (1.0 - sc.shadow * np.exp(-xx / (0.05 * w)))[..., None]
    if i in sc.coffee:
        cx, cy, r = w * rng.uniform(0.55, 0.75), h * rng.uniform(0.28, 0.4), w * 0.11  # over the body text
        dist = np.hypot((xx - cx) / 1.0, (yy - cy) / 0.93)
        ring = np.exp(-(((dist - r) / (0.012 * w)) ** 2)) * 0.55 + (dist < r) * 0.10
        stain = np.array([0.55, 0.38, 0.18], np.float32)
        a = a * (1.0 - ring[..., None] * (1.0 - stain))
    if sc.copier:
        edge = np.maximum(np.exp(-(w - xx) / (0.012 * w)), np.exp(-(h - yy) / (0.01 * h)))
        a = a * (1.0 - 0.85 * edge)[..., None]
        for fy in (0.28, 0.72):
            hole = np.hypot(xx - 0.045 * w, yy - fy * h) < 0.016 * w
            a[hole] = 30.0
    if sc.speckle:
        for _ in range(sc.speckle):
            x, y, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.6, 2.2) * s / 4
            a[(xx - x) ** 2 + (yy - y) ** 2 < r * r] = 40.0
    img = _blend(img, a)
    img = img.filter(ImageFilter.GaussianBlur(sc.blur * sc.dpi / 300))
    a = np.asarray(img, np.float32)
    noise = rng.normal(0.0, sc.noise, size=a.shape[:2])[..., None]
    img = _blend(img, a + noise)
    if not sc.colour:
        img = img.convert("L")
    if sc.fax:
        # Fax "fine" mode: half the vertical resolution, one bit per pixel.
        half = img.convert("L").resize((img.width, img.height // 2), Image.Resampling.BOX)
        bits = np.asarray(half, np.float32) + rng.normal(0, 14, size=(half.height, half.width))
        img = Image.fromarray(np.where(bits > 150, 255, 0).astype(np.uint8)).resize(img.size, Image.Resampling.NEAREST)
        img = img.convert("1", dither=Image.Dither.NONE)
    return img


def _scan(clean: list[Image.Image], spec: DocSpec, key: _Key) -> list[Image.Image]:
    rng = np.random.default_rng(spec.seed)
    out = []
    for i, img in enumerate(clean):
        back = clean[i + 1] if i + 1 < len(clean) else (clean[i - 1] if i else None)
        out.append(_scan_page(img, back, spec, i, key, rng))
    return out


def _pdf(pages: list[Image.Image], dpi: int, jpeg: int) -> bytes:
    pdf = pikepdf.new()
    for img in pages:
        w_pt, h_pt = img.width * 72 / dpi, img.height * 72 / dpi
        if img.mode == "1":
            image = pikepdf.Stream(pdf, zlib.compress(img.tobytes()))
            image.Filter = pikepdf.Name.FlateDecode
            image.BitsPerComponent = 1
            image.ColorSpace = pikepdf.Name.DeviceGray
        else:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg)
            image = pikepdf.Stream(pdf, buf.getvalue())
            image.Filter = pikepdf.Name.DCTDecode
            image.BitsPerComponent = 8
            image.ColorSpace = pikepdf.Name.DeviceGray if img.mode == "L" else pikepdf.Name.DeviceRGB
        image.Type = pikepdf.Name.XObject
        image.Subtype = pikepdf.Name.Image
        image.Width, image.Height = img.size
        content = pikepdf.Stream(pdf, f"q {w_pt:.3f} 0 0 {h_pt:.3f} 0 0 cm /Im0 Do Q".encode())
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page,
            MediaBox=[0, 0, w_pt, h_pt],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=image)),
            Contents=content,
        )
        pdf.pages.append(pikepdf.Page(page))
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()


@dataclass
class Built:
    spec: DocSpec
    pdf: bytes
    items: list[Item] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)


def build(spec: DocSpec) -> Built:
    key = _Key()
    pages = _scan(_typeset(spec, key), spec, key)
    return Built(spec, _pdf(pages, spec.scan.dpi, spec.scan.jpeg), key.items, key.regions)


# -- answer keys ---------------------------------------------------------------------------------


def _grouped(items: list[Item]) -> list[dict]:
    groups: dict[tuple[str, str, str], dict] = {}
    for it in items:
        g = groups.setdefault(
            (it.action, it.kind, it.text),
            {"kind": it.kind, "text": it.text, "action": it.action, "count": 0, "pages": [], "where": []},
        )
        g["count"] += 1
        if it.page not in g["pages"]:
            g["pages"].append(it.page)
        if it.where not in g["where"]:
            g["where"].append(it.where)
    return list(groups.values())


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def answer_markdown(b: Built) -> str:
    groups = _grouped(b.items)
    replaced = [g for g in groups if g["action"] != "keep"]
    kept = [g for g in groups if g["action"] == "keep"]
    out = [
        f"# {b.spec.title}",
        "",
        f"- File: `{b.spec.slug}.pdf` ({len(b.spec.pages)} pages, image only)",
        f"- Tier: **{b.spec.tier}**",
        "- Policy: client profile",
        "",
        "## What makes it hard",
        "",
    ]
    out += [f"- {c}" for c in b.spec.challenges]
    out += ["", f"## Must be replaced ({len(replaced)})", "", "| Kind | Value | Pages | Where |", "|---|---|---|---|"]
    out += [
        f"| {g['kind']} | {_md_cell(g['text'])} | {', '.join(map(str, g['pages']))} | {', '.join(g['where'])} |"
        for g in replaced
    ]
    out += ["", f"## Must stay as it is ({len(kept)})", "", "| Kind | Value | Pages |", "|---|---|---|"]
    out += [f"| {g['kind']} | {_md_cell(g['text'])} | {', '.join(map(str, g['pages']))} |" for g in kept]
    out += ["", f"## Picture areas ({len(b.regions)})", "", "| Page | Kind | Tool should | Note |", "|---|---|---|---|"]
    out += [f"| {r.page} | {r.kind} | {r.action} | {_md_cell(r.note)} |" for r in b.regions]
    return "\n".join(out) + "\n"


def answer_json(b: Built) -> str:
    scan = asdict(b.spec.scan)
    return (
        json.dumps(
            {
                "document": f"{b.spec.slug}.pdf",
                "tier": b.spec.tier,
                "title": b.spec.title,
                "pages": len(b.spec.pages),
                "profile": "client",
                "challenges": b.spec.challenges,
                "scan": scan,
                # Region boxes are on the typeset page; these say how each page was turned before scanning.
                "page_layout": [{"landscape": p.landscape, "upside_down": p.upside_down} for p in b.spec.pages],
                "values": _grouped(b.items),
                "regions": [asdict(r) for r in b.regions],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def _index(built: list[Built]) -> str:
    out = [
        "# Synthetic scanned medico-legal corpus",
        "",
        "All people, organisations, numbers and addresses are fictitious (phone numbers from the ACMA "
        "ranges reserved for fiction). Each PDF has a `.answers.md` (for reading) and `.answers.json` "
        "(for scripts) listing what the client profile must replace, what must stay, and the picture areas.",
        "",
        "| Document | Tier | Pages | Must replace | Must stay | Picture areas |",
        "|---|---|---|---|---|---|",
    ]
    for b in built:
        groups = _grouped(b.items)
        n_rep = sum(g["action"] != "keep" for g in groups)
        out.append(
            f"| [{b.spec.title}]({b.spec.slug}.answers.md) | {b.spec.tier} | {len(b.spec.pages)} | {n_rep} | "
            f"{len(groups) - n_rep} | {len(b.regions)} |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    from synthdocs.corpus_docs import DOCS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("out/synth_corpus"))
    ap.add_argument("--only", nargs="*", help="slugs to build (default: all)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    built = []
    for spec in DOCS:
        if args.only and spec.slug not in args.only:
            continue
        b = build(spec)
        (args.out / f"{spec.slug}.pdf").write_bytes(b.pdf)
        (args.out / f"{spec.slug}.answers.md").write_text(answer_markdown(b), "utf-8")
        (args.out / f"{spec.slug}.answers.json").write_text(answer_json(b), "utf-8")
        built.append(b)
        print(f"{spec.slug}: {len(spec.pages)} pages, {len(b.pdf) // 1024} KB")  # noqa: T201 - CLI progress
    if not args.only:
        (args.out / "README.md").write_text(_index(built), "utf-8")


if __name__ == "__main__":
    main()
