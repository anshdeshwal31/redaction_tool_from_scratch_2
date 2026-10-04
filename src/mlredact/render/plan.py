"""Render planning: decisions -> per-page drawing operations in canonical raster pixels.

Coverage is always whole tokens, unioned per line, padded by max(pad_min_px, frac * line height)
and clipped to the page.  Multi-line mentions become one operation per line; for surrogates each
line segment receives the surrogate tokens aligned with the original tokens on that line.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

from mlredact.config.schema import RedactionRenderConfig
from mlredact.core.canonical import content_id
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.geometry import BBox
from mlredact.core.model import Line, PageGeometry, RenderOp, Token
from mlredact.core.types import ActionKind
from mlredact.policy.engine import Decision
from mlredact.surrogate.assign import MentionSurrogate

# Raster operations this build can execute.  Planning fails closed for anything else.
SUPPORTED_KINDS = frozenset(
    {
        ActionKind.BLACKOUT,
        ActionKind.REMOVE,
        ActionKind.SURROGATE,
        ActionKind.LABEL,
        ActionKind.SUPPRESS_INK,
        ActionKind.RETYPESET_REGION,
    }
)
_TEXT_KINDS = frozenset({ActionKind.SURROGATE, ActionKind.LABEL})
_DESCENDERS = frozenset("gjpqy,;()[]{}_")


def effective_kind(action: ActionKind, style: str) -> ActionKind:
    if action in (ActionKind.KEEP, ActionKind.SUPPRESS_INK, ActionKind.REMOVE, ActionKind.RETYPESET_REGION):
        return action
    return {"blackout": ActionKind.BLACKOUT, "label": ActionKind.LABEL, "surrogate": ActionKind.SURROGATE}[style]


def check_supported(style: str, actions: Sequence[ActionKind]) -> None:
    for action in actions:
        kind = effective_kind(action, style)
        if kind is not ActionKind.KEEP and kind not in SUPPORTED_KINDS:
            raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID, action=kind)


def rotated_token_ids(lines: Sequence[Line], max_degrees: float = 25.0) -> frozenset[str]:
    """Tokens of lines whose baseline (quad top edge) is more than ``max_degrees`` from horizontal."""
    out: set[str] = set()
    for line in lines:
        (x0, y0), (x1, y1) = line.quad[0], line.quad[1]
        angle = abs(math.degrees(math.atan2(y1 - y0, x1 - x0)))
        if min(angle, 180.0 - angle) > max_degrees:
            out.update(t.token_id for t in line.tokens)
    return frozenset(out)


def pad_box(box: BBox, line_height: int, cfg: RedactionRenderConfig, page: BBox) -> BBox:
    pad = max(cfg.pad_min_px, round(cfg.pad_line_height_frac * line_height))
    return box.pad(pad, pad, page)


def plan_ops(
    decisions: Sequence[Decision],
    tokens: Mapping[str, Token],
    geometries: Mapping[int, PageGeometry],
    cfg: RedactionRenderConfig,
    surrogates: Mapping[str, MentionSurrogate] | None = None,
    rotated_tokens: AbstractSet[str] = frozenset(),
    lines: Sequence[Line] = (),
) -> dict[int, tuple[RenderOp, ...]]:
    """``rotated_tokens``: tokens on lines far from horizontal (sideways pages, vertical table
    headers).  Text is only typeset horizontally, so their mentions are blacked out instead.
    ``lines``: the page lines, needed for line re-typesetting (dense blocks, positional hardening)."""
    ops: dict[int, list[RenderOp]] = defaultdict(list)
    keyed: list[tuple[tuple[int, int], RenderOp]] = []
    surrogates = surrogates or {}
    for d in decisions:
        if d.action is ActionKind.KEEP:
            continue
        kind = effective_kind(d.action, cfg.style)
        m = d.mention
        if kind in _TEXT_KINDS and any(t in rotated_tokens for t in m.token_ids):
            kind = ActionKind.BLACKOUT
        sur = surrogates.get(m.mention_id) if kind in _TEXT_KINDS else None
        if kind is ActionKind.SURROGATE and sur is None:
            kind = ActionKind.BLACKOUT  # no surrogate could be produced: fail safe, never keep
        per_line: dict[tuple[int, int], BBox] = {}
        per_line_text: dict[tuple[int, int], list[str]] = defaultdict(list)
        per_line_orig: dict[tuple[int, int], list[str]] = defaultdict(list)
        for i, tid in enumerate(m.token_ids):
            tok = tokens[tid]
            key = (tok.page, tok.line_no)
            per_line[key] = per_line[key].union(tok.bbox) if key in per_line else tok.bbox
            per_line_orig[key].append(tok.text)
            if sur is not None and sur.tokens[i]:
                per_line_text[key].append(sur.tokens[i])
        for key, box in sorted(per_line.items()):
            page = key[0]
            padded = pad_box(box, box.height, cfg, geometries[page].bounds)
            if padded.is_empty:
                continue
            keyed.append(
                (
                    key,
                    RenderOp(
                        page=page,
                        kind=kind,
                        box=padded,
                        reason=m.entity_type.value,
                        ref_id=m.mention_id,
                        text=" ".join(per_line_text[key]) if kind in _TEXT_KINDS else None,
                        ink_box=box,
                        has_descender=any(c in _DESCENDERS for c in "".join(per_line_orig[key])),
                        source_text=" ".join(per_line_orig[key]) if kind in _TEXT_KINDS else None,
                    ),
                )
            )
    if lines and cfg.style != "blackout":
        keyed = _retypeset_lines(keyed, decisions, surrogates, lines, geometries, cfg, rotated_tokens)
    for _key, op in keyed:
        ops[op.page].append(op)
    return {p: tuple(sorted(v, key=lambda o: o.sort_key)) for p, v in sorted(ops.items())}


def _blocks(lines: Sequence[Line]) -> list[list[Line]]:
    """Consecutive lines of a page that sit tightly under each other with overlapping columns."""
    out: list[list[Line]] = []
    for line in sorted(lines, key=lambda ln: (ln.page, ln.bbox.y0, ln.bbox.x0, ln.line_no)):
        if out:
            prev = out[-1][-1]
            gap = line.bbox.y0 - prev.bbox.y1
            overlap = min(line.bbox.x1, prev.bbox.x1) - max(line.bbox.x0, prev.bbox.x0)
            narrow = min(line.bbox.width, prev.bbox.width)
            if (
                line.page == prev.page
                and -0.3 * prev.bbox.height <= gap <= 0.8 * prev.bbox.height
                and overlap >= 0.3 * narrow
            ):
                out[-1].append(line)
                continue
        out.append([line])
    return out


def _retypeset_lines(
    keyed: list[tuple[tuple[int, int], RenderOp]],
    decisions: Sequence[Decision],
    surrogates: Mapping[str, MentionSurrogate],
    lines: Sequence[Line],
    geometries: Mapping[int, PageGeometry],
    cfg: RedactionRenderConfig,
    rotated_tokens: AbstractSet[str],
) -> list[tuple[tuple[int, int], RenderOp]]:
    """Replace per-mention text ops by whole-line ops where the line must be re-typeset."""
    replacement: dict[str, str] = {}  # token id -> text drawn in its place ("" = absorbed)
    mention_of: dict[str, str] = {}
    for d in decisions:
        sur = surrogates.get(d.mention.mention_id)
        if d.action is ActionKind.KEEP or sur is None:
            continue
        for i, tid in enumerate(d.mention.token_ids):
            replacement[tid] = sur.tokens[i]
            mention_of[tid] = d.mention.mention_id
    by_key: dict[tuple[int, int], list[RenderOp]] = defaultdict(list)
    for key, op in keyed:
        by_key[key].append(op)

    def eligible(line: Line) -> bool:
        line_ops = by_key.get((line.page, line.line_no), [])
        return (
            bool(line_ops)
            and all(op.kind in _TEXT_KINDS for op in line_ops)
            and not any(t.token_id in rotated_tokens for t in line.tokens)
        )

    chosen: set[tuple[int, int]] = set()
    if cfg.positional_hardening == "line_retypeset":
        chosen.update((ln.page, ln.line_no) for ln in lines if eligible(ln))
    if cfg.dense_blocks:
        for block in _blocks(lines):
            if len(block) > cfg.dense_block_max_lines:
                continue
            toks = [t for ln in block for t in ln.tokens]
            pii = sum(t.token_id in mention_of for t in toks)
            mentions = {mention_of[t.token_id] for t in toks if t.token_id in mention_of}
            frac = pii / max(1, len(toks))
            if frac >= cfg.dense_block_pii_fraction or (len(mentions) >= 2 and frac >= 0.25):
                chosen.update((ln.page, ln.line_no) for ln in block if eligible(ln))
    if not chosen:
        return keyed
    out = [(key, op) for key, op in keyed if key not in chosen]
    for line in lines:
        key = (line.page, line.line_no)
        if key not in chosen:
            continue
        words = [replacement.get(t.token_id, t.text) for t in line.tokens]
        text = " ".join(w for w in words if w)
        source = " ".join(t.text for t in line.tokens)
        ink = line.tokens[0].bbox
        for t in line.tokens[1:]:
            ink = ink.union(t.bbox)
        box = pad_box(ink, ink.height, cfg, geometries[line.page].bounds)
        covers = tuple(sorted({mention_of[t.token_id] for t in line.tokens if t.token_id in mention_of}))
        out.append(
            (
                key,
                RenderOp(
                    page=line.page,
                    kind=ActionKind.RETYPESET_REGION,
                    box=box,
                    reason="retypeset",
                    ref_id=content_id("L", line.page, line.line_no, *box.as_tuple()),
                    text=text,
                    ink_box=ink,
                    has_descender=any(c in _DESCENDERS for c in source),
                    source_text=source,
                    covers=covers,
                ),
            )
        )
    return out
