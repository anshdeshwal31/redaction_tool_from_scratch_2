"""OCR engine A: PP-OCRv6 (DB detection + CTC recognition) on ONNX Runtime.

RapidOCR supplies the detector pre/post-processing and the text-line orientation classifier; we
create every ONNX Runtime session ourselves (fixed threads, deterministic kernels, no downloads),
force batch size 1 everywhere (results independent of the other lines on the page), and run
recognition + CTC decoding ourselves to obtain character positions and alternative readings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from mlredact.config.schema import OcrEngineAConfig, ThreadsConfig
from mlredact.core.geometry import BBox, quad_from_array
from mlredact.core.model import Line, Token, token_id
from mlredact.ocr.ctc import DecodedChar, decode_ctc, token_alternates
from mlredact.registry.models import load_registry, resolve
from mlredact.runtime.determinism import cpu_providers, ort_session_options

_REC_HEIGHT = 48
_REC_MIN_WIDTH = 320
_CLS_THRESHOLD = 0.9


class _Cfg(dict[str, Any]):
    """dict with attribute access, as RapidOCR components expect."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


@dataclass(frozen=True, slots=True)
class _Crop:
    image: npt.NDArray[np.uint8]
    inverse: npt.NDArray[np.float64]  # warped-crop px -> page px (3x3 homography)
    width: int
    height: int
    rotated90: bool


def _warp_crop(img: npt.NDArray[np.uint8], quad: npt.NDArray[np.float32]) -> _Crop | None:
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    cw = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    ch = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if cw < 2 or ch < 2:
        return None
    std = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, std)
    dst = cv2.warpPerspective(img, m, (cw, ch), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    rotated90 = ch / cw >= 1.5
    if rotated90:
        dst = np.rot90(dst)
    return _Crop(np.ascontiguousarray(dst), np.linalg.inv(m.astype(np.float64)), cw, ch, rotated90)


def _map_points(inverse: npt.NDArray[np.float64], pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for x, y in pts:
        vx, vy, vz = inverse @ np.array([x, y, 1.0], dtype=np.float64)
        out.append((float(vx / vz), float(vy / vz)))
    return out


class PPOCREngine:
    engine_id = "ppocrv6"

    def __init__(self, cfg: OcrEngineAConfig, threads: ThreadsConfig, models_dir: Path) -> None:
        import onnxruntime as ort
        from rapidocr.ch_ppocr_cls import TextClassifier
        from rapidocr.ch_ppocr_det import TextDetector
        from rapidocr.utils.typings import EngineType, OCRVersion

        self._cfg = cfg
        registry = load_registry()
        det_path = next(iter(resolve(cfg.det_model, models_dir).values()))
        rec_path = next(iter(resolve(cfg.rec_model, models_dir).values()))
        self.version = "+".join(
            f"{mid}@{registry[mid].files[0].sha256[:12]}"
            for mid in (cfg.det_model, cfg.rec_model, cfg.cls_model)
            if mid != cfg.cls_model or cfg.use_textline_orientation
        )

        def session(path: Path) -> Any:
            return ort.InferenceSession(str(path), sess_options=ort_session_options(threads), providers=cpu_providers())

        self._det = TextDetector(
            _Cfg(
                engine_type=EngineType.ONNXRUNTIME,
                session=session(det_path),
                limit_side_len=cfg.det_limit_side_len,
                limit_type="min",
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                thresh=cfg.det_thresh,
                box_thresh=cfg.det_box_thresh,
                max_candidates=2000,
                unclip_ratio=cfg.det_unclip_ratio,
                use_dilation=True,
                score_mode="fast",
            )
        )
        self._cls: Any = None
        if cfg.use_textline_orientation:
            cls_path = next(iter(resolve(cfg.cls_model, models_dir).values()))
            self._cls = TextClassifier(
                _Cfg(
                    engine_type=EngineType.ONNXRUNTIME,
                    session=session(cls_path),
                    ocr_version=OCRVersion.PPOCRV5,
                    cls_batch_num=1,
                    cls_thresh=_CLS_THRESHOLD,
                    label_list=["0", "180"],
                )
            )
        self._rec = session(rec_path)
        self._rec_input = self._rec.get_inputs()[0].name
        characters = self._rec.get_modelmeta().custom_metadata_map["character"].splitlines()
        self._charset = ["<blank>", *characters, " "]

    # -------------------------------------------------------------------------------- recognition
    def _recognise(self, img: npt.NDArray[np.uint8]) -> tuple[list[DecodedChar], float]:
        h, w = img.shape[:2]
        ratio = w / float(h)
        max_wh_ratio = max(_REC_MIN_WIDTH / _REC_HEIGHT, ratio)
        input_w = int(_REC_HEIGHT * max_wh_ratio)
        resized_w = input_w if math.ceil(_REC_HEIGHT * ratio) > input_w else math.ceil(_REC_HEIGHT * ratio)
        resized = cv2.resize(img, (resized_w, _REC_HEIGHT)).astype(np.float32)
        tensor = resized.transpose((2, 0, 1)) / 255.0
        tensor = (tensor - 0.5) / 0.5
        padded = np.zeros((3, _REC_HEIGHT, input_w), dtype=np.float32)
        padded[:, :, :resized_w] = tensor
        probs = self._rec.run(None, {self._rec_input: padded[np.newaxis]})[0][0]
        decoded = decode_ctc(
            probs,
            self._charset,
            input_w,
            top_k=self._cfg.alternates_top_k,
            min_alt_prob=self._cfg.alternate_min_prob,
        )
        return list(decoded.chars), w / float(resized_w)

    # ------------------------------------------------------------------------------------ public
    def ocr_page(self, page_rgb: npt.NDArray[np.uint8], page_index: int) -> tuple[Line, ...]:
        bgr = np.asarray(cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        page_h, page_w = bgr.shape[:2]
        bounds = BBox(0, 0, page_w, page_h)
        det = self._det(bgr)
        if det.boxes is None or len(det.boxes) == 0:
            return ()
        lines: list[Line] = []
        for line_no, quad in enumerate(det.boxes):
            crop = _warp_crop(bgr, quad)
            if crop is None:
                continue
            img = crop.image
            rotated180 = False
            if self._cls is not None:
                cls_out = self._cls([img])
                label, score = cls_out.cls_res[0]
                if "180" in label and score > _CLS_THRESHOLD:
                    rotated180 = True
                    img = np.ascontiguousarray(cls_out.img_list[0])
            chars, scale = self._recognise(img)
            line_box = BBox.from_points(quad_from_array(quad)).intersection(bounds)
            if line_box is None:
                continue
            tokens = self._build_tokens(
                chars, scale, img.shape[1], img.shape[0], crop, rotated180, line_box, bounds, page_index, line_no
            )
            if not tokens:
                continue
            letters = [c.prob for c in chars if not c.char.isspace()]
            lines.append(
                Line(
                    line_id=f"p{page_index:04d}.l{line_no:04d}",
                    page=page_index,
                    line_no=line_no,
                    bbox=line_box,
                    quad=quad_from_array(quad),
                    confidence=round(float(np.mean(letters)), 5) if letters else 0.0,
                    engine=self.engine_id,
                    tokens=tuple(tokens),
                )
            )
        return tuple(lines)

    def _build_tokens(
        self,
        chars: list[DecodedChar],
        scale: float,
        final_w: int,
        final_h: int,
        crop: _Crop,
        rotated180: bool,
        line_box: BBox,
        bounds: BBox,
        page_index: int,
        line_no: int,
    ) -> list[Token]:
        # Group characters into whitespace-separated tokens (positions in final-crop pixels).
        groups: list[list[tuple[DecodedChar, float]]] = []
        current: list[tuple[DecodedChar, float]] = []
        for c in chars:
            if c.char.isspace():
                if current:
                    groups.append(current)
                    current = []
                continue
            current.append((c, c.x_center * scale))
        if current:
            groups.append(current)
        tokens: list[Token] = []
        for word_no, group in enumerate(groups):
            # Tokens tile the line: boundaries sit midway between neighbouring tokens' edge chars,
            # and the outer tokens extend to the crop edges, so every ink pixel of the line belongs
            # to some token box.
            left = 0.0 if word_no == 0 else (groups[word_no - 1][-1][1] + group[0][1]) / 2.0
            right = float(final_w) if word_no == len(groups) - 1 else (group[-1][1] + groups[word_no + 1][0][1]) / 2.0
            if crop.rotated90:
                box = line_box  # vertical text: conservative, the whole line
            else:
                x0, x1 = left, right
                if rotated180:
                    x0, x1 = final_w - right, final_w - left
                corners = [(x0, 0.0), (x1, 0.0), (x1, float(final_h)), (x0, float(final_h))]
                mapped = BBox.from_points(_map_points(crop.inverse, corners)).intersection(bounds)
                box = mapped if mapped is not None else line_box
            probs = [c.prob for c, _ in group]
            text = "".join(c.char for c, _ in group)
            tokens.append(
                Token(
                    token_id=token_id(page_index, line_no, word_no),
                    page=page_index,
                    line_no=line_no,
                    word_no=word_no,
                    bbox=box,
                    confidence=round(float(np.mean(probs)), 5),
                    min_confidence=round(float(min(probs)), 5),
                    engine=self.engine_id,
                    text=text,
                    alternates=token_alternates([c for c, _ in group], self._cfg.max_alternates_per_token),
                )
            )
        return tokens
