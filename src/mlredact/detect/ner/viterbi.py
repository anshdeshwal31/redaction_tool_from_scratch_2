"""Constrained BIOES decoding for token classifiers (plan §7.2 N1).

Label ids follow the model's ``id2label`` (``O``, ``B-cat``, ``I-cat``, ``E-cat``, ``S-cat``).  Only
well-formed paths are allowed:

* ``O``, ``E-x`` and ``S-x`` may be followed by ``O``, ``B-y`` or ``S-y``;
* ``B-x`` and ``I-x`` may be followed by ``I-x`` or ``E-x`` (same category);
* a path starts with ``O``/``B``/``S`` and ends with ``O``/``E``/``S``.

Each allowed transition carries one of six published bias names, so the operating point is a
configuration value.  The decoder is exact (max-sum over the whole document, float64) and breaks ties
by the lowest label id, so the path is a deterministic function of the logits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mlredact.config.schema import ViterbiBiases

_NEG = -1e30


@dataclass(frozen=True, slots=True)
class TagSet:
    kinds: tuple[str, ...]  # per label id: "O", "B", "I", "E", "S"
    cats: tuple[int, ...]  # per label id: category index, -1 for O
    categories: tuple[str, ...]

    @property
    def background(self) -> int:
        return self.kinds.index("O")

    @classmethod
    def from_id2label(cls, id2label: Mapping[int, str]) -> TagSet:
        ids = sorted(id2label)
        if ids != list(range(len(ids))):
            raise ValueError("label ids must be contiguous from 0")
        kinds: list[str] = []
        cats: list[int] = []
        categories: list[str] = []
        for i in ids:
            label = id2label[i]
            if label == "O":
                kinds.append("O")
                cats.append(-1)
                continue
            kind, _, cat = label.partition("-")
            if kind not in {"B", "I", "E", "S"} or not cat:
                raise ValueError(f"not a BIOES label: {label!r}")
            if cat not in categories:
                categories.append(cat)
            kinds.append(kind)
            cats.append(categories.index(cat))
        if kinds.count("O") != 1:
            raise ValueError("exactly one background label is required")
        return cls(tuple(kinds), tuple(cats), tuple(categories))


def transitions(
    tags: TagSet, biases: ViterbiBiases
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(start, trans, end)`` score arrays; disallowed moves score ``-1e30``."""
    k = len(tags.kinds)
    start = np.full(k, _NEG)
    end = np.full(k, _NEG)
    trans = np.full((k, k), _NEG)
    for j, kind in enumerate(tags.kinds):
        if kind in {"O", "B", "S"}:
            start[j] = 0.0
        if kind in {"O", "E", "S"}:
            end[j] = 0.0
    for i, (ki, ci) in enumerate(zip(tags.kinds, tags.cats, strict=True)):
        for j, (kj, cj) in enumerate(zip(tags.kinds, tags.cats, strict=True)):
            if ki == "O":
                if kj == "O":
                    trans[i, j] = biases.background_stay
                elif kj in {"B", "S"}:
                    trans[i, j] = biases.background_to_start
            elif ki in {"E", "S"}:
                if kj == "O":
                    trans[i, j] = biases.end_to_background
                elif kj in {"B", "S"}:
                    trans[i, j] = biases.end_to_start
            elif ci == cj:  # ki in B, I: stay inside the same category
                if kj == "I":
                    trans[i, j] = biases.inside_to_continue
                elif kj == "E":
                    trans[i, j] = biases.inside_to_end
    return start, trans, end


def log_softmax(logits: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def viterbi(
    emissions: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    trans: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
) -> list[int]:
    """Best label path for ``emissions`` ``(N, K)`` (log-probabilities)."""
    n = emissions.shape[0]
    if n == 0:
        return []
    score = start + emissions[0]
    back = np.zeros((n, emissions.shape[1]), dtype=np.int64)
    for t in range(1, n):
        cand = score[:, None] + trans  # (prev, next)
        back[t] = cand.argmax(axis=0)  # argmax returns the first (lowest id) maximum: deterministic
        score = cand[back[t], np.arange(cand.shape[1])] + emissions[t]
    last = int((score + end).argmax())
    path = [last]
    for t in range(n - 1, 0, -1):
        last = int(back[t, last])
        path.append(last)
    path.reverse()
    return path


def path_spans(path: list[int], tags: TagSet) -> list[tuple[int, int, int]]:
    """``(first, last, category)`` token spans (inclusive) of a well-formed BIOES path."""
    spans: list[tuple[int, int, int]] = []
    open_at: int | None = None
    for t, label in enumerate(path):
        kind, cat = tags.kinds[label], tags.cats[label]
        if kind == "S":
            spans.append((t, t, cat))
            open_at = None
        elif kind == "B":
            open_at = t
        elif kind == "E" and open_at is not None:
            spans.append((open_at, t, cat))
            open_at = None
        elif kind == "O":
            open_at = None
    return spans


def posterior_runs(
    probs: npt.NDArray[np.float64], tags: TagSet, threshold: float, covered: npt.NDArray[np.bool_]
) -> list[tuple[int, int, int, float]]:
    """Recall sweep: maximal runs of uncovered tokens with ``P(not O) >= threshold``.

    Returns ``(first, last, category, mean P(not O))``; the category is the one with the largest
    total probability mass over the run (ties: lowest category index).
    """
    p_entity = 1.0 - probs[:, tags.background]
    hot = (p_entity >= threshold) & ~covered
    cats = np.array(tags.cats)
    out: list[tuple[int, int, int, float]] = []
    t = 0
    n = len(hot)
    while t < n:
        if not hot[t]:
            t += 1
            continue
        first = t
        while t + 1 < n and hot[t + 1]:
            t += 1
        mass = np.zeros(len(tags.categories))
        for c in range(len(tags.categories)):
            mass[c] = probs[first : t + 1, cats == c].sum()
        out.append((first, t, int(mass.argmax()), float(p_entity[first : t + 1].mean())))
        t += 1
    return out
