"""Deterministic page-image encoding (pinned libjpeg-turbo / zlib via the runtime profile)."""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

from mlredact.config.schema import OutputConfig


@dataclass(frozen=True, slots=True)
class EncodedImage:
    data: bytes
    width: int
    height: int
    colorspace: str  # "DeviceRGB" | "DeviceGray"
    filter: str  # "DCTDecode" | "FlateDecode"


def is_gray(rgb: npt.NDArray[np.uint8]) -> bool:
    return bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))


def encode_raster(rgb: npt.NDArray[np.uint8], cfg: OutputConfig) -> EncodedImage:
    gray = is_gray(rgb)
    arr = np.ascontiguousarray(rgb[..., 0] if gray else rgb)
    height, width = arr.shape[:2]
    colorspace = "DeviceGray" if gray else "DeviceRGB"
    if cfg.image_encoding == "flate":
        return EncodedImage(zlib.compress(arr.tobytes(), 6), width, height, colorspace, "FlateDecode")
    img = Image.fromarray(arr, mode="L" if gray else "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=cfg.jpeg_quality, subsampling=0, optimize=False, progressive=False)
    return EncodedImage(buf.getvalue(), width, height, colorspace, "DCTDecode")


def decode_image(enc: EncodedImage) -> npt.NDArray[np.uint8]:
    if enc.filter == "FlateDecode":
        channels = 1 if enc.colorspace == "DeviceGray" else 3
        flat = np.frombuffer(zlib.decompress(enc.data), dtype=np.uint8)
        return flat.reshape(enc.height, enc.width, channels) if channels == 3 else flat.reshape(enc.height, enc.width)
    with Image.open(io.BytesIO(enc.data)) as img:
        return np.asarray(img.convert("L" if enc.colorspace == "DeviceGray" else "RGB"), dtype=np.uint8)
