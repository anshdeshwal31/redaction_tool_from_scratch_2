"""N1: OpenAI Privacy Filter (Apache-2.0) on ONNX Runtime with our own constrained decoder.

* The whole view is tokenised once (no special tokens); character offsets map tokens back to text.
* Logits are computed in fixed windows of ``window_tokens``.  Attention is banded, so a token's
  logits depend only on tokens within ``context_tokens`` (layers x band) of it; each window
  contributes only its *core*, the positions with full context inside the window (or that touch a
  document edge).  The stitched logits therefore equal whole-document inference, whatever the length.
* One constrained BIOES Viterbi pass runs over the whole document (biases from configuration).
* A posterior sweep adds weak candidates where the model put enough mass on "not background" but
  the Viterbi path did not (recall first).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mlredact.config.schema import PrivacyFilterConfig, ThreadsConfig
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.model import Candidate, Evidence
from mlredact.core.types import EvidenceStrength, ViewKind
from mlredact.detect.allowlist import refine_span
from mlredact.detect.ner.viterbi import TagSet, log_softmax, path_spans, posterior_runs, transitions, viterbi
from mlredact.registry.models import load_registry, resolve
from mlredact.runtime.determinism import cpu_providers, ort_session_options
from mlredact.text.views import TextView

DETECTOR_ID = "ner.privacy_filter"


def core_windows(n: int, window: int, context: int) -> list[tuple[int, int, int, int]]:
    """``(win_start, win_end, core_start, core_end)`` tiling ``[0, n)`` with exact cores.

    Cores are disjoint and cover every position once; each core position has ``context`` tokens of
    context on both sides inside its window, unless that side is a document edge.
    """
    if window <= 2 * context:
        raise ValueError("window must exceed twice the context")
    if n <= window:
        return [(0, n, 0, n)] if n > 0 else []
    out: list[tuple[int, int, int, int]] = []
    core_start = 0
    win_start = 0
    while core_start < n:
        win_end = min(n, win_start + window)
        core_end = win_end if win_end == n else win_end - context
        out.append((win_start, win_end, core_start, core_end))
        core_start = core_end
        win_start = core_start - context
    return out


class PrivacyFilterDetector:
    detector_id = DETECTOR_ID

    def __init__(self, cfg: PrivacyFilterConfig, threads: ThreadsConfig, models_dir: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._cfg = cfg
        paths = resolve(cfg.model, models_dir)
        entry = load_registry()[cfg.model]
        self.version = f"{cfg.model}@{entry.files[0].sha256[:12]}"
        model_cfg: dict[str, Any] = json.loads(paths["config.json"].read_text("utf-8"))
        self._tags = TagSet.from_id2label({int(k): v for k, v in model_cfg["id2label"].items()})
        derived = int(model_cfg["num_hidden_layers"]) * int(model_cfg["sliding_window"])
        self._context = cfg.context_tokens if cfg.context_tokens is not None else derived
        if cfg.window_tokens <= 2 * self._context:
            raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
        self._start, self._trans, self._end = transitions(self._tags, cfg.biases)
        self._tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        onnx_name = next(name for name in sorted(paths) if name.endswith(".onnx"))
        # External weights (``*.onnx_data``) are resolved next to the graph; both are hash-verified.
        self._session = ort.InferenceSession(
            str(paths[onnx_name]), sess_options=ort_session_options(threads), providers=cpu_providers()
        )

    # ------------------------------------------------------------------------------- inference
    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids and character offsets (no special tokens)."""
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        return list(enc.ids), [(int(a), int(b)) for a, b in enc.offsets]

    def logits(self, ids: list[int], window_tokens: int | None = None) -> npt.NDArray[np.float32]:
        """Whole-document logits ``(N, labels)``, stitched from exact window cores.

        ``window_tokens`` overrides the configured window (used to verify that stitching is exact).
        """
        n = len(ids)
        out = np.zeros((n, len(self._tags.kinds)), dtype=np.float32)
        arr = np.asarray(ids, dtype=np.int64)
        window = window_tokens or self._cfg.window_tokens
        for ws, we, cs, ce in core_windows(n, max(window, 2 * self._context + 1), self._context):
            x = arr[ws:we][None]
            win = self._session.run(None, {"input_ids": x, "attention_mask": np.ones_like(x)})[0][0]
            out[cs:ce] = win[cs - ws : ce - ws]
        return out

    def decode(self, logits: npt.NDArray[Any]) -> list[tuple[int, int, int, float, bool]]:
        """``(first, last, category, score, from_viterbi)`` token spans."""
        logp = log_softmax(logits)
        path = viterbi(logp, self._start, self._trans, self._end)
        probs = np.exp(logp)
        spans: list[tuple[int, int, int, float, bool]] = []
        covered = np.zeros(len(path), dtype=bool)
        for first, last, cat in path_spans(path, self._tags):
            score = float(np.mean([probs[t, path[t]] for t in range(first, last + 1)]))
            spans.append((first, last, cat, score, True))
            covered[first : last + 1] = True
        if self._cfg.posterior_threshold < 1.0:
            for first, last, cat, score in posterior_runs(probs, self._tags, self._cfg.posterior_threshold, covered):
                spans.append((first, last, cat, score, False))
        return sorted(spans)

    # ---------------------------------------------------------------------------------- detect
    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        out: list[Candidate] = []
        for kind in self._cfg.views:
            view = views[kind]
            text = view.text
            ids, offsets = self.tokenize(text)
            if not ids:
                continue
            for first, last, cat, score, from_viterbi in self.decode(self.logits(ids)):
                category = self._tags.categories[cat]
                etype = self._cfg.categories.get(category)
                if etype is None:
                    continue
                start, end = offsets[first][0], offsets[last][1]
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                rng = view.token_range(start, end)
                refined = refine_span(view, rng.start, rng.stop - 1, etype) if len(rng) else None
                if refined is None:
                    continue
                start, end, etype = view.starts[refined[0]], view.ends[refined[1]], refined[2]
                strong = from_viterbi and score >= self._cfg.strong_threshold
                out.append(
                    Candidate(
                        entity_type=etype,
                        view=kind,
                        start=start,
                        end=end,
                        evidence=Evidence(
                            DETECTOR_ID,
                            self.version,
                            f"pf.{category}" + ("" if from_viterbi else ".posterior"),
                            EvidenceStrength.STRONG if strong else EvidenceStrength.WEAK,
                            round(score, 4),
                        ),
                    )
                )
        return out
