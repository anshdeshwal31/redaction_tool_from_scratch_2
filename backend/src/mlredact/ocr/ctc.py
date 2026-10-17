"""CTC decoding with character positions, confidences and alternative readings.

The stock decoders return only the best string.  For redaction we also need:

* each character's horizontal position (to build word boxes that tile the line), and
* per-character runner-up readings (e.g. ``0``/``O``, ``1``/``l``) so OCR-tolerant detectors and
  propagation can match identifiers and names the top reading got wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

BLANK_INDEX = 0


@dataclass(frozen=True, slots=True)
class DecodedChar:
    char: str = field(repr=False)
    x_center: float  # in *input tensor* pixels (before mapping back to the crop)
    prob: float
    # (char, prob) runner-ups at the emission step, most likely first; blank excluded.
    alternatives: tuple[tuple[str, float], ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class DecodedLine:
    chars: tuple[DecodedChar, ...]

    @property
    def text(self) -> str:
        return "".join(c.char for c in self.chars)


def decode_ctc(
    probs: npt.NDArray[np.float32],
    charset: list[str],
    input_width: int,
    *,
    top_k: int = 2,
    min_alt_prob: float = 0.05,
) -> DecodedLine:
    """Greedy CTC decode of one sequence ``probs`` of shape ``(T, C)``.

    ``charset[i]`` is the symbol for class ``i`` (``charset[0]`` is the blank).  ``input_width`` is
    the padded tensor width used for inference; time step ``t`` covers
    ``[t, t+1) * input_width / T`` pixels.
    """
    if probs.ndim != 2:
        raise ValueError("expected a (T, C) probability matrix")
    steps = probs.shape[0]
    if steps == 0:
        return DecodedLine(())
    best = probs.argmax(axis=1)
    step_width = input_width / steps
    chars: list[DecodedChar] = []
    t = 0
    while t < steps:
        idx = int(best[t])
        run_end = t + 1
        while run_end < steps and int(best[run_end]) == idx:
            run_end += 1
        if idx != BLANK_INDEX:
            # Emission step = most confident step of the run (deterministic: first maximum).
            run_probs = probs[t:run_end, idx]
            local = int(np.argmax(run_probs))
            step = t + local
            prob = float(run_probs[local])
            alternatives: list[tuple[str, float]] = []
            if top_k > 1:
                row = probs[step]
                k = min(top_k + 1, row.shape[0])
                # argpartition is deterministic for a given input; then order the few candidates
                # by (-prob, class index) so ties resolve identically everywhere.
                cand = np.argpartition(-row, k - 1)[:k]
                for cls in sorted((int(c) for c in cand), key=lambda c: (-float(row[c]), c)):
                    if cls in (idx, BLANK_INDEX):
                        continue
                    p = float(row[cls])
                    if p < min_alt_prob or len(alternatives) >= top_k - 1:
                        continue
                    alternatives.append((charset[cls], round(p, 5)))
            centre = (t + run_end) / 2.0 * step_width
            chars.append(DecodedChar(charset[idx], centre, round(prob, 5), tuple(alternatives)))
        t = run_end
    return DecodedLine(tuple(chars))


def token_alternates(chars: list[DecodedChar], max_alternates: int) -> tuple[str, ...]:
    """Alternative readings of a token: single substitutions ranked by runner-up probability,
    then the all-runner-ups reading.  Deterministic order; the original reading is excluded."""
    if max_alternates <= 0:
        return ()
    base = [c.char for c in chars]
    subs: list[tuple[float, int, str]] = []
    for i, c in enumerate(chars):
        for alt, p in c.alternatives:
            subs.append((-p, i, alt))
    subs.sort()
    out: list[str] = []
    seen = {"".join(base)}
    for _neg_p, i, alt in subs:
        cand = base.copy()
        cand[i] = alt
        s = "".join(cand)
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_alternates:
            return tuple(out)
    if subs:
        best_per_position: dict[int, str] = {}
        for _neg_p, i, alt in subs:  # already sorted by descending probability
            best_per_position.setdefault(i, alt)
        combo = base.copy()
        for i, alt in best_per_position.items():
            combo[i] = alt
        s = "".join(combo)
        if s not in seen and len(out) < max_alternates:
            out.append(s)
    return tuple(out)
