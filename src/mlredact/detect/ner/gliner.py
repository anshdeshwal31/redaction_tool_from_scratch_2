"""N2: GLiNER-PII (Knowledgator, Apache-2.0) zero-shot NER on ONNX Runtime, without PyTorch.

Reimplements the inference path of ``gliner`` 0.2.x exactly:

* words: regex ``\\w+(?:[-_]\\w+)*|\\S`` (the "whitespace" splitter), with character offsets;
* prompt: ``<<ENT>> label ... <<SEP>>`` prepended as words, tokenised with ``is_pretokenized``;
* ``words_mask``: 1-based index of each text word on its first sub-token (prompt words skipped);
* span models (``markerV0``): every span ``(i, i + w)`` with ``w < max_width``, ``span_mask`` marks
  those inside the text; logits ``(B, L, W, C)``; sigmoid above threshold;
* token models (``token_level``): logits ``(B, L, C, 3)`` = start / end / inside; a span needs start
  and end above threshold for the same class and every inside score above it; its score is the
  minimum of those;
* flat greedy selection by descending score (ties keep the deterministic enumeration order).

Text is processed in fixed, overlapping word windows with batch size 1, so a window's result never
depends on document length or on any other input.  Overlapping windows are unioned (recall first):
a name cut by one window's edge is seen whole by the next.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mlredact.config.schema import GlinerConfig, ThreadsConfig
from mlredact.core.model import Candidate, Evidence
from mlredact.core.types import EvidenceStrength, ViewKind
from mlredact.detect.allowlist import refine_span
from mlredact.registry.models import load_registry, resolve
from mlredact.runtime.determinism import cpu_providers, ort_session_options
from mlredact.text.views import TextView

DETECTOR_ID = "ner.gliner"
_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")
# Safety net only: windows are sized well below this.  A longer window is split in two (with overlap).
_MAX_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class Span:
    start: int  # word index (inclusive)
    end: int  # word index (inclusive)
    label: int
    score: float


def split_words(text: str) -> list[tuple[int, int]]:
    """GLiNER's whitespace word splitter, as ``(start, end)`` character offsets."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def word_windows(n_words: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Fixed ``[start, end)`` word windows covering ``n_words`` with the given overlap."""
    if n_words <= 0:
        return []
    step = window - overlap
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(n_words, start + window)
        out.append((start, end))
        if end >= n_words:
            return out
        start += step


def greedy_flat(spans: Sequence[Span]) -> list[Span]:
    """gliner's flat-NER greedy search: highest score first, no two spans share a word."""
    chosen: list[Span] = []
    for s in sorted(spans, key=lambda x: -x.score):  # stable sort: ties keep enumeration order
        if all(s.end < c.start or c.end < s.start for c in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda x: (x.start, x.end, x.label))


def _sigmoid(x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def decode_span_logits(logits: npt.NDArray[Any], n_words: int, threshold: float) -> list[Span]:
    """Span models: ``logits`` is ``(L, W, C)`` for one text of ``n_words`` words."""
    probs = _sigmoid(logits[:n_words])
    spans = [
        Span(int(s), int(s + k), int(c), float(probs[s, k, c]))
        for s, k, c in zip(*np.nonzero(probs > threshold), strict=True)
        if s + k + 1 <= n_words
    ]
    return greedy_flat(spans)


def decode_token_logits(logits: npt.NDArray[Any], n_words: int, threshold: float) -> list[Span]:
    """Token models: ``logits`` is ``(L, C, 3)`` (start, end, inside) for one text."""
    probs = _sigmoid(logits[:n_words])
    start, end, inside = probs[..., 0], probs[..., 1], probs[..., 2]
    starts = list(zip(*np.nonzero(start > threshold), strict=True))
    ends = list(zip(*np.nonzero(end > threshold), strict=True))
    spans: list[Span] = []
    for st, c in starts:
        for ed, c_end in ends:
            if ed < st or c_end != c:
                continue
            ins = inside[st : ed + 1, c]
            if bool((ins < threshold).any()):
                continue
            score = min(float(ins.min()), float(start[st, c]), float(end[ed, c]))
            spans.append(Span(int(st), int(ed), int(c), score))
    return greedy_flat(spans)


class GlinerDetector:
    detector_id = DETECTOR_ID

    def __init__(self, cfg: GlinerConfig, threads: ThreadsConfig, models_dir: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._cfg = cfg
        paths = resolve(cfg.model, models_dir)
        entry = load_registry()[cfg.model]
        self.version = f"{cfg.model}@{entry.files[0].sha256[:12]}"
        gcfg: dict[str, Any] = json.loads(paths["gliner_config.json"].read_text("utf-8"))
        self._token_mode = gcfg.get("span_mode") == "token_level"
        self._max_width = int(gcfg.get("max_width", 12))
        self._tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        onnx_name = next(name for name in sorted(paths) if name.endswith(".onnx"))
        self._session = ort.InferenceSession(
            str(paths[onnx_name]), sess_options=ort_session_options(threads), providers=cpu_providers()
        )
        self._inputs = {i.name: i.type for i in self._session.get_inputs()}
        self._labels = sorted(cfg.labels)
        self._types = [cfg.labels[label] for label in self._labels]
        prompt: list[str] = []
        for label in self._labels:
            prompt += [str(gcfg.get("ent_token", "<<ENT>>")), label]
        prompt.append(str(gcfg.get("sep_token", "<<SEP>>")))
        self._prompt = prompt

    # ------------------------------------------------------------------------------- inference
    def _encode(self, words: Sequence[str]) -> tuple[list[int], list[int], list[int]]:
        enc = self._tokenizer.encode([*self._prompt, *words], is_pretokenized=True)
        skip = len(self._prompt)
        mask: list[int] = []
        prev: int | None = None
        seen = 0
        for wid in enc.word_ids:
            if wid is None:
                mask.append(0)
            else:
                first = wid != prev
                if first:
                    seen += 1
                mask.append(seen - skip if (first and seen > skip) else 0)
            prev = wid
        return list(enc.ids), list(enc.attention_mask), mask

    def infer(self, words: Sequence[str]) -> list[Span]:
        """Spans (word indices) for one window of words."""
        if not words:
            return []
        ids, attention, words_mask = self._encode(words)
        if len(ids) > _MAX_TOKENS and len(words) > 1:
            half = len(words) // 2
            overlap = min(16, half)
            left = self.infer(words[: half + overlap])
            shift = half - overlap
            right = [Span(s.start + shift, s.end + shift, s.label, s.score) for s in self.infer(words[shift:])]
            return [*left, *right]
        n = len(words)
        feeds: dict[str, npt.NDArray[Any]] = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([attention], dtype=np.int64),
            "words_mask": np.array([words_mask], dtype=np.int64),
            "text_lengths": np.array([[n]], dtype=np.int64),
        }
        if "span_idx" in self._inputs:
            starts = np.repeat(np.arange(n, dtype=np.int64), self._max_width)
            widths = np.tile(np.arange(self._max_width, dtype=np.int64), n)
            span_idx = np.stack([starts, starts + widths], axis=1)
            mask = span_idx[:, 1] < n
            feeds["span_idx"] = span_idx[None]
            feeds["span_mask"] = mask[None] if "bool" in self._inputs["span_mask"] else mask[None].astype(np.int64)
        logits = self._session.run(None, {k: v for k, v in feeds.items() if k in self._inputs})[0][0]
        if self._token_mode:
            return decode_token_logits(logits, n, self._cfg.threshold)
        return decode_span_logits(logits, n, self._cfg.threshold)

    # ---------------------------------------------------------------------------------- detect
    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        out: list[Candidate] = []
        done: list[str] = []  # texts of views already processed
        for kind in self._cfg.views:
            view = views[kind]
            # Views are length-preserving case/character transforms, so word boundaries are identical
            # in every view; a window whose text equals an earlier view's gives identical results.
            words = split_words(view.text)
            seen: set[tuple[int, int, int]] = set()
            for w0, w1 in word_windows(len(words), self._cfg.window_words, self._cfg.overlap_words):
                chunk = words[w0:w1]
                a, b = chunk[0][0], chunk[-1][1]
                if any(text[a:b] == view.text[a:b] for text in done):
                    continue
                for span in self.infer([view.text[s:e] for s, e in chunk]):
                    key = (w0 + span.start, w0 + span.end, span.label)
                    if key in seen:
                        continue
                    seen.add(key)
                    etype = self._types[span.label]
                    rng = view.token_range(chunk[span.start][0], chunk[span.end][1])
                    refined = refine_span(view, rng.start, rng.stop - 1, etype) if len(rng) else None
                    if refined is None:
                        continue
                    start, end, etype = view.starts[refined[0]], view.ends[refined[1]], refined[2]
                    strong = span.score >= self._cfg.type_strong_thresholds.get(etype, self._cfg.strong_threshold)
                    out.append(
                        Candidate(
                            entity_type=etype,
                            view=kind,
                            start=start,
                            end=end,
                            evidence=Evidence(
                                DETECTOR_ID,
                                self.version,
                                "gliner." + self._labels[span.label].replace(" ", "_"),
                                EvidenceStrength.STRONG if strong else EvidenceStrength.WEAK,
                                round(span.score, 4),
                            ),
                        )
                    )
            done.append(view.text)
        return out
