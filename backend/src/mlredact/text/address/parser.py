"""The parser: the chain of stages run in order, and the conversion of its working units into the public result."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from mlredact.text.address.internal import KeptB, UnitB
from mlredact.text.address.model import AddressParse, AddressUnit, AddrLine, BoxRole, KeptPlace, LabelSpan
from mlredact.text.address.sweep import FinalStage
from mlredact.text.address.words import bare, pc_value
from mlredact.text.gazetteer import Gazetteer, load_gazetteer


class Parser(FinalStage):
    """One parse of one document.  Build it with the lines, call :meth:`run` once."""

    def run(self) -> AddressParse:
        self.labels = {ln.li: self.labels_in(ln) for ln in self.L}
        self.anchors = {ln.li: self.anchors_in(ln) for ln in self.L}
        for s in self.streets():
            self.join_street(s)
        self.form_units()
        self.place_only_units()
        self.repeat_sweep()
        self.lowercase_lines()
        self.record_kept()
        for u in self.units:
            self.finish_unit(u)
            u.form = all(self.L[li].box is BoxRole.VALUE for li in {p[0] for p in u.roles})
        ordered = sorted(self.units, key=lambda u: min(u.roles))
        self._link_repeats(ordered)
        uids = {id(u): f"U{i}" for i, u in enumerate(ordered, 1)}
        return AddressParse(
            tuple(_public(u, uids) for u in ordered),
            tuple(_kept(k) for k in sorted(self.kept, key=lambda k: min(k.positions))),
            tuple(self.notes),
            tuple(
                LabelSpan(li, lab.start, lab.end, lab.role.value, lab.colon, lab.whole)
                for li in sorted(self.labels)
                for lab in self.labels[li]
            ),
        )

    @staticmethod
    def _link_repeats(ordered: list[UnitB]) -> None:
        """A unit whose street part and place agree with an earlier unit's repeats it (found by the join or by the
        repeat sweep)."""
        first: dict[tuple[str, tuple[tuple[str, ...], ...]], UnitB] = {}
        for u in ordered:
            if u.repeat_src is not None or not u.key.street_kind:
                continue
            k = (u.key.street_kind, u.key.street_parts)
            src = first.get(k)
            if src is not None and src.key.same_address(u.key):
                u.repeat_src = src
            else:
                first.setdefault(k, u)


def _public(u: UnitB, uids: dict[int, str]) -> AddressUnit:
    return AddressUnit(
        uid=uids[id(u)],
        kind=u.kind,
        roles=MappingProxyType({p: u.roles[p] for p in sorted(u.roles)}),
        street=tuple(u.street),
        locality=tuple(u.locality),
        how=tuple(u.how),
        partial=tuple(u.partial),
        key=u.key,
        repeat_of=uids.get(id(u.repeat_src)) if u.repeat_src is not None else None,
        overseas=u.overseas,
        form=u.form,
        form_parts=tuple(u.form_parts),
    )


def _kept(k: KeptB) -> KeptPlace:
    return KeptPlace(k.kind, tuple(sorted(k.positions)), k.after_cue)


def parse_document(lines: Sequence[AddrLine], gazetteer: Gazetteer | None = None) -> AddressParse:
    """Find the address units of a document.

    ``lines`` are the document's lines in reading order, with their geometry (see :class:`AddrLine`); positions in the
    result are ``(line index, word index)`` into this sequence.  ``gazetteer`` defaults to the packaged locality list.
    Pure and deterministic: the same lines give the same result."""
    return Parser(lines, gazetteer or load_gazetteer()).run()


def parse_texts(texts: Sequence[str], gazetteer: Gazetteer | None = None) -> AddressParse:
    """Find the address units in plain lines of text laid out as a single column (one line per entry, no gaps).  For
    callers that have words but no geometry, such as one detected mention."""
    return parse_document([AddrLine.from_text(t, row=i) for i, t in enumerate(texts)], gazetteer)


def parse_mention(words: Sequence[str], gazetteer: Gazetteer | None = None) -> AddressUnit | None:
    """Read the words of one detected address mention (one line) and return the unit that covers most of them, or
    None.  ``words`` are non-empty and hold no whitespace; positions in the returned unit are ``(0, word index)``.
    For a caller that has a mention, not a document: the detector's candidate, a policy split, a fake."""
    parse = parse_document([AddrLine.from_text(" ".join(words))], gazetteer)
    if not parse.units:
        return None
    return max(parse.units, key=lambda u: len(u.roles))


def place_start(words: Sequence[str], gazetteer: Gazetteer | None = None) -> int | None:
    """The index of the first word of the place (suburb, state, postcode) of an address mention, or None when the
    grammar finds no place in it.  0 for a place-only mention ('Parramatta NSW 2150')."""
    unit = parse_mention(words, gazetteer)
    if unit is None or not unit.locality:
        return None
    return unit.locality[0][1]


def postcode_value(word: str) -> int | None:
    """The postcode a single word spells (four digits, with OCR letters O, I, l, S and B allowed when three digits are
    real), ignoring brackets, quotes and trailing punctuation around it; None for any other word.  The word alone is
    judged, not its place: 'postcode 2750' and the 'NSW 2750' of an address both give 2750."""
    return pc_value(bare(word))
