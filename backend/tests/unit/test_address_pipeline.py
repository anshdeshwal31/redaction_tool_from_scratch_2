"""Whole addresses through the real runner (intake, join, fakes, planning, rendering, checks V1-V6, re-plans), with the
in-process harness of test_bench_runner: the worker tasks run in this process and page analysis reads the PDF's own
text layer instead of running OCR, so no models are needed.  The documents are born-digital PDFs laid out from grid
lines (fictional text only)."""

from __future__ import annotations

import dataclasses
import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pikepdf
import pytest
from test_bench_runner import NO_MODELS, InProcessPool

import mlredact.pipeline.runner as runner_module
from mlredact.config.loader import load_config
from mlredact.core.model import Mention
from mlredact.core.types import ActionKind, JobStatus
from mlredact.pipeline.runner import JobResult, Redactor
from mlredact.policy.addresses import AddressJoin, MentionLayout, UnitPart
from mlredact.verify.needles import tokens_of
from support import GridLine, column, pdf_from_grid

FOOTER_ROW = 52  # the bottom 5% of the page: 60 + 52 * 14.4 pt of 842


@dataclass
class Run:
    result: JobResult
    pool: InProcessPool

    @property
    def checks(self) -> dict[str, dict[str, Any]]:
        return {c["check"]: c for c in self.result.manifest["verification"]["checks"]}

    @property
    def failed(self) -> list[str]:
        return sorted(k for k, c in self.checks.items() if not c["passed"])

    @property
    def blackouts(self) -> list[tuple[str, str]]:
        trace = self.result.trace
        assert trace is not None
        return [
            (op.reason, op.ref_id)
            for ops in trace.ops.values()
            for op in ops
            if op.kind is ActionKind.BLACKOUT and op.reason != "signature"
        ]

    @property
    def replans(self) -> list[Any]:
        trace = self.result.trace
        assert trace is not None
        return [op for ops in trace.ops.values() for op in ops if op.ref_id.startswith("R-")]

    def words(self) -> list[str]:
        """The words the output's text layer holds (every string the builder wrote), as tokens."""
        pdf = self.result.pdf or self.result.held_back_pdf
        assert pdf is not None
        out: list[str] = []
        with pikepdf.open(io.BytesIO(pdf)) as doc:
            for page in doc.pages:
                content = page.obj.Contents.read_bytes()
                for raw in re.findall(rb"\(((?:[^()\\]|\\.)*)\) Tj", content):
                    out.extend(tokens_of(raw.decode("cp1252")))
        return out

    def tokens(self) -> list[str]:
        return [w for w in self.words() if w != "~"]


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Run]:
    def go(
        lines: Sequence[GridLine],
        profile: str = "client",
        overrides: dict[str, Any] | None = None,
        join: Callable[[Callable[..., AddressJoin]], Callable[..., AddressJoin]] | None = None,
    ) -> Run:
        pool = InProcessPool()
        pool.install(monkeypatch)
        if join is not None:
            monkeypatch.setattr(runner_module, "join_addresses", join(runner_module.join_addresses))
        config = load_config(
            profile, (), {**NO_MODELS, "surrogate": {"allow_development_key": True}, **(overrides or {})}
        )
        with Redactor(config) as redactor:
            result = redactor.run(pdf_from_grid(lines), keep_held_back=True, trace=True)
        return Run(result, pool)

    return go


def released_clean(r: Run) -> None:
    assert r.result.status is JobStatus.RELEASED, (r.result.manifest["reasons"], r.failed)
    assert r.blackouts == [] and r.replans == [], "a black box or a re-plan where the fakes should stand"


# ------------------------------------------------------------------------------------------------ whole addresses
class TestWholeAddresses:
    def test_one_line_is_one_unit_and_a_kept_place_in_the_same_suburb_stays_readable(
        self, run: Callable[..., Run]
    ) -> None:
        r = run(
            column("Ms Jane Citizen", "Address: 42 Wattle Grove Road, Penrith NSW 2750", "", "She grew up in Penrith.")
        )
        released_clean(r)
        words = r.tokens()
        assert not {"wattle", "grove", "2750"} & set(words)
        assert words.count("penrith") == 1, "the place on its own stays, the address's is replaced"
        assert r.checks["V2_text"]["metrics"]["address_needles"] > 0 and r.checks["V5_surrogate"]["passed"]
        assert r.result.trace is not None and len(r.result.trace.units) == 1

    def test_a_block_is_replaced_line_by_line(self, run: Callable[..., Run]) -> None:
        r = run(column("Ms Jane Citizen", "12 Smith Street", "PENRITH NSW 2750", "", "Thank you for your letter."))
        released_clean(r)
        assert not {"smith", "penrith", "2750"} & set(r.tokens())
        trace = r.result.trace
        assert trace is not None and len(trace.units) == 1
        (fake,) = trace.address_fakes.values()
        assert sum(1 for v in fake.values() if v) >= 4
        assert "nsw" in r.tokens()  # the state word stays (address_state: keep)

    def test_an_address_over_a_page_break_is_one_unit(self, run: Callable[..., Run]) -> None:
        lines = [
            GridLine("Please reply to", 0, 0),
            GridLine("12 Smith Street", 0, 1),
            GridLine("PENRITH NSW 2750", 1, 0),
            GridLine("Thank you", 1, 1),
        ]
        r = run(lines)
        released_clean(r)
        assert not {"smith", "penrith", "2750"} & set(r.tokens())
        assert r.result.trace is not None and len(r.result.trace.units) == 1

    def test_a_form_with_kept_dates_amounts_and_places_is_released_without_a_black_box(
        self, run: Callable[..., Run]
    ) -> None:
        """One document: a three-box form (postcode 2021), a kept date holding the digits 2021, a kept place, an amount
        and a clinical score.  Nothing is held and nothing is blacked out."""
        r = run(
            column(
                "Residential address: 40 Bridge Street",
                "Suburb/Town Paddington",
                "State NSW",
                "Postcode 2021",
                "",
                "Date of report: 14/03/2021",
                "She grew up in Paddington and later lived in Penrith.",
                "The fee was $12,750.00.",
                "Pain level 4 of 10.",
            )
        )
        released_clean(r)
        words = r.tokens()
        assert not {"bridge"} & set(words)
        for kept in ("14", "03", "2021", "12", "750", "pain", "level", "4", "10", "penrith"):
            assert kept in words, kept
        assert words.count("paddington") == 1  # the place on its own; the form's is replaced
        trace = r.result.trace
        assert trace is not None and len(trace.units) == 1
        (unit,) = trace.units.values()
        assert unit.form and len(unit.mention_ids) >= 2

    def test_a_footer_repeat_gets_the_letterhead_fake(self, run: Callable[..., Run]) -> None:
        """The footer names the street and the capital but no state or postcode: only the repeat sweep finds it."""
        r = run(WITH_FOOTER)
        released_clean(r)
        words = r.tokens()
        assert not {"george", "123", "2000"} & set(words)
        assert words.count("sydney") == 1, "only the place on its own stays"
        trace = r.result.trace
        assert trace is not None and len(trace.units) == 2
        first, second = trace.units.values()
        assert second.repeat_of == first.unit_id
        assert trace.address_fakes[first.unit_id]  # one fake for the key, two places it is drawn

    def test_a_place_only_line_is_replaced_and_a_place_after_a_cue_is_kept(self, run: Callable[..., Run]) -> None:
        r = run(
            column(
                "Ms Jane Citizen",
                "PENRITH NSW 2750",
                "",
                "Date of accident: 12 June 2023, Parramatta NSW 2150",
            )
        )
        released_clean(r)
        words = r.tokens()
        assert "penrith" not in words and "2750" not in words
        assert {"parramatta", "nsw", "2150"} <= set(words)

    def test_an_overseas_address_is_replaced_whole_and_keeps_its_country(self, run: Callable[..., Run]) -> None:
        r = run(column("She now lives at 12 Queen Street, Auckland 1010, New Zealand."))
        released_clean(r)
        words = r.tokens()
        assert not {"queen", "auckland", "1010"} & set(words)
        assert {"new", "zealand"} <= set(words)

    @pytest.mark.parametrize("profile", ["broad", "claimant_focused", "maximal"])
    def test_every_profile_replaces_the_whole_address(self, run: Callable[..., Run], profile: str) -> None:
        r = run(column("Ms Jane Citizen", "12 Smith Street", "PENRITH NSW 2750", "", "Thank you."), profile)
        assert r.result.status is JobStatus.RELEASED, r.failed
        assert not {"smith", "penrith", "2750"} & set(r.tokens())
        assert r.blackouts == []

    def test_parts_mode_is_todays_behaviour(self, run: Callable[..., Run]) -> None:
        r = run(
            column("Address: 42 Wattle Grove Road, Penrith NSW 2750"),
            overrides={"policy": {"addresses": "parts"}},
        )
        assert r.result.status is JobStatus.RELEASED
        trace = r.result.trace
        assert trace is not None and not trace.units and not trace.layouts and not trace.address_fakes
        words = r.tokens()
        assert "wattle" not in words and "penrith" in words  # the locality is kept: Joel's literal answer
        assert "address_needles" not in r.checks["V2_text"]["metrics"]

    def test_the_trace_records_the_join(self, run: Callable[..., Run]) -> None:
        r = run(column("Address: 42 Wattle Grove Road, Penrith NSW 2750", "", "She grew up in Penrith."))
        trace = r.result.trace
        assert trace is not None
        assert len(trace.units) == 1 and trace.layouts and len(trace.address_fakes) == 1
        assert {k.kind.value for k in trace.kept_places} == {"repeat_in_prose"}
        assert set(trace.layouts) <= {d.mention.mention_id for d in trace.decisions}


# -------------------------------------------------------------------------------- what verification catches
def street_line_only(real: Callable[..., AddressJoin]) -> Callable[..., AddressJoin]:
    """A join that replaces only the street line of each unit: its place is left behind."""

    def join(mentions: Sequence[Mention], pages: Any, policy: Any, *args: Any, **kwargs: Any) -> AddressJoin:
        result = real(mentions, pages, policy, *args, **kwargs)
        kept: list[Mention] = []
        layouts = dict(result.layouts)
        for m in result.mentions:
            layout = layouts.get(m.mention_id)
            if layout is None or layout.place_start is None:
                kept.append(m)
                continue
            ids = m.token_ids[: layout.place_start]
            if not ids:
                del layouts[m.mention_id]
                continue
            kept.append(dataclasses.replace(m, token_ids=ids, value=m.value.split(" ", len(ids))[0]))
            layouts[m.mention_id] = MentionLayout(
                layout.unit_id, UnitPart.STREET, None, layout.roles[: layout.place_start]
            )
        return dataclasses.replace(result, mentions=tuple(kept), layouts=layouts)

    return join


def without_the_sweep(real: Callable[..., AddressJoin]) -> Callable[..., AddressJoin]:
    """A join that misses the repeats of an address (no repeat sweep)."""

    def join(mentions: Sequence[Mention], pages: Any, policy: Any, *args: Any, **kwargs: Any) -> AddressJoin:
        return real(mentions, pages, policy, *args, **{**kwargs, "repeat_sweep": False})

    return join


NO_REPLAN: dict[str, Any] = {"verification": {"max_replans": 0}}
LABELLED = column("Ms Jane Citizen", "Address: 40 Bridge Street, Penrith NSW 2750", "", "Thank you.")
WITH_FOOTER = [
    *column(
        "Smith & Jones Lawyers",
        "Level 4, 123 George Street, Sydney NSW 2000",
        "",
        "Our Sydney office will help.",
    ),
    GridLine("Smith & Jones Lawyers | 123 George Street Sydney | (02) 5550 1234", 0, FOOTER_ROW),
]


class TestVerificationCatches:
    def test_a_left_over_place_line_fails_v2_and_v4(self, run: Callable[..., Run]) -> None:
        r = run(LABELLED, overrides=NO_REPLAN, join=street_line_only)
        assert r.result.status is JobStatus.QUARANTINED and r.result.pdf is None
        assert r.failed == ["V2_text", "V4_residual"]
        assert r.checks["V2_text"]["metrics"]["address_hits"] >= 1
        assert r.checks["V4_residual"]["metrics"]["from_removed_values"] >= 1
        assert {"penrith", "2750"} <= set(r.tokens()), "the held-back copy still shows the place"

    def test_only_the_street_line_replaced_is_a_residual_not_explained_by_the_fake(
        self, run: Callable[..., Run]
    ) -> None:
        """'Address: 40 Bridge Street, Penrith NSW 2750' with only the street line replaced: the mention that the
        rules find again contains the fake street line, which no longer explains it."""
        r = run(LABELLED, overrides=NO_REPLAN, join=street_line_only)
        assert r.checks["V4_residual"]["metrics"]["from_redetection"] >= 1 or r.failed

    def test_a_re_plan_replaces_the_left_over_place_and_the_job_is_released(self, run: Callable[..., Run]) -> None:
        r = run(LABELLED, join=street_line_only)
        assert r.result.status is JobStatus.RELEASED, r.failed
        assert len(r.replans) >= 1
        words = r.tokens()
        assert "penrith" not in words and "2750" not in words

    def test_a_footer_repeat_the_join_missed_is_caught(self, run: Callable[..., Run]) -> None:
        r = run(WITH_FOOTER, overrides=NO_REPLAN, join=without_the_sweep)
        assert r.result.status is JobStatus.QUARANTINED
        assert r.failed == ["V2_text", "V4_residual"]

    def test_a_missed_repeat_is_re_planned_with_the_fake_of_its_unit(self, run: Callable[..., Run]) -> None:
        r = run(WITH_FOOTER, join=without_the_sweep)
        assert r.result.status is JobStatus.RELEASED, r.failed
        assert r.blackouts == [], "the repeat takes the fake of the address it repeats, not a black box"
        assert r.replans and all(op.kind is ActionKind.SURROGATE and op.text for op in r.replans)
        words = r.tokens()
        assert not {"king", "50", "2000"} & set(words)
        trace = r.result.trace
        assert trace is not None
        (fake,) = trace.address_fakes.values()
        fake_words = {w for text in fake.values() for w in tokens_of(text)}
        drawn = {w for op in r.replans for w in tokens_of(op.text or "")}
        assert drawn and drawn <= fake_words, "the same fake words as the letterhead's"

    def test_the_exemptions_hold_a_kept_place_in_the_unit_suburb_is_not_a_hit(self, run: Callable[..., Run]) -> None:
        r = run(
            column("Address: 12 Smith Street, Penrith NSW 2750", "", "She grew up in Penrith."), overrides=NO_REPLAN
        )
        assert r.result.status is JobStatus.RELEASED, r.failed
        assert r.checks["V2_text"]["metrics"]["address_hits"] == 0
        assert r.checks["V4_residual"]["metrics"]["residual_tokens"] == 0


# ------------------------------------------------------------------------------------------------ detector fixes
class TestDetectorFixes:
    """Over-spans and look-alikes of the rule detectors that used to surface as re-plan black boxes or held jobs
    (steps 2 and 4 of the build).  Each is released with no black box and no re-plan, in the profile named."""

    @pytest.mark.parametrize("profile", ["client", "broad"])
    def test_rsd_in_brackets_is_no_box_and_nothing_is_blacked_out(self, run: Callable[..., Run], profile: str) -> None:
        r = run(column("Diagnosis: CRPS (RSD 2) of the right hand."), profile)
        released_clean(r)
        assert {"crps", "rsd", "2", "hand"} <= set(r.tokens())

    def test_a_standalone_postcode_over_a_text_layer_without_spaces_is_not_blacked_out(
        self, run: Callable[..., Run]
    ) -> None:
        """Broad replaces a postcode on its own; the re-read output holds '2070weresurveyed.' as one word, and the
        postcode label must not take it for a postcode."""
        r = run(column("Residents of postcode 2750 were surveyed."), "broad")
        released_clean(r)
        words = r.tokens()
        assert "2750" not in words and {"residents", "postcode", "surveyed"} <= set(words)
        kept = run(column("Residents of postcode 2750 were surveyed."), "client")
        released_clean(kept)
        assert "2750" in kept.tokens()

    @pytest.mark.parametrize("profile", ["client", "broad"])
    def test_form_boxes_on_their_own_lines_are_one_unit_and_the_labels_stay(
        self, run: Callable[..., Run], profile: str
    ) -> None:
        r = run(
            column(
                "Address:",
                "Unit",
                "5",
                "Street No.",
                "12",
                "Street name",
                "Smith Street",
                "Suburb/Town",
                "Penrith",
                "State",
                "NSW",
                "Postcode",
                "2750",
            ),
            profile,
        )
        released_clean(r)
        words = r.tokens()
        assert not {"smith", "penrith", "2750", "12"} & set(words)
        assert {"address", "unit", "street", "suburb", "town", "state", "nsw", "postcode"} <= set(words)
        assert r.result.trace is not None and len(r.result.trace.units) == 1

    @pytest.mark.parametrize("profile", ["client", "broad", "claimant_focused", "maximal"])
    def test_an_address_followed_by_a_phone_number_is_one_unit_and_one_phone(
        self, run: Callable[..., Run], profile: str
    ) -> None:
        r = run(column("Ms Jane Citizen", "Address: 5 Smith Street, Penrith NSW 2750 T (02) 5550 6610"), profile)
        released_clean(r)
        words = r.tokens()
        assert not {"smith", "penrith", "2750", "5550", "6610"} & set(words)
        assert "t" in words, "the label of the phone stays"
        trace = r.result.trace
        assert trace is not None and len(trace.units) == 1
        phones = [d for d in trace.decisions if d.mention.entity_type.value == "phone"]
        assert len(phones) == 1 and phones[0].mention.value == "(02) 5550 6610"

    def test_a_state_label_keeps_its_value_and_is_never_a_locality(self, run: Callable[..., Run]) -> None:
        r = run(column("Suburb: Penrith State: NSW", "", "State: NSW"), "broad")
        released_clean(r)
        words = r.tokens()
        assert words.count("nsw") == 2 and "penrith" not in words
        trace = r.result.trace
        assert trace is not None
        assert [d.mention.value for d in trace.decisions if d.mention.entity_type.value == "locality"] == ["Penrith"]

    @pytest.mark.parametrize("profile", ["client", "claimant_focused"])
    def test_a_surname_that_is_a_street_name_does_not_split_the_unit(
        self, run: Callable[..., Run], profile: str
    ) -> None:
        """'Smith' in '12 Smith Street' used to make the whole address a PERSON mention (a person outranks an address
        where they overlap): replaced with a person's name, or kept for a clinician under claimant_focused."""
        r = run(
            column("Dr John Smith, GP", "Address: 12 Smith Street, Penrith NSW 2750", "Dr Smith reviewed the claim."),
            profile,
        )
        released_clean(r)
        trace = r.result.trace
        assert trace is not None and len(trace.units) == 1
        (unit,) = trace.units.values()
        assert unit.excluded == (), "no mention owns a word of the unit"
        assert not {"penrith", "2750"} & set(r.tokens())
        types = [d.mention.entity_type.value for d in trace.decisions]
        assert types.count("street_address") == 1 and types.count("person") == 2

    def test_dr_after_a_street_name_is_the_street_type_and_the_unit_is_whole(self, run: Callable[..., Run]) -> None:
        r = run(column("45 Smith Dr Penrith NSW 2750"), "client")
        released_clean(r)
        assert not {"smith", "penrith", "2750"} & set(r.tokens())
        trace = r.result.trace
        assert trace is not None and len(trace.units) == 1
        (unit,) = trace.units.values()
        assert unit.excluded == ()

    def test_a_guard_look_alike_in_a_letter_is_left_alone_in_every_profile(self, run: Callable[..., Run]) -> None:
        lines = column(
            "The 6 Minute Walk Test was 410 m.",
            "He completed 30 Second Chair Rise twice.",
            "3 Local Court convictions were recorded.",
            "He developed RSD 3 months after the injury.",
        )
        for profile in ("client", "broad", "claimant_focused", "maximal"):
            r = run(lines, profile)
            released_clean(r)
            words = r.tokens()
            assert {"6", "minute", "walk", "test", "30", "second", "chair", "rise", "3", "local", "court"} <= set(words)
            assert {"rsd", "3", "months"} <= set(words)

    @pytest.mark.parametrize("profile", ["client", "broad"])
    def test_a_state_box_is_part_of_its_unit_and_a_lone_state_label_is_not(
        self, run: Callable[..., Run], profile: str
    ) -> None:
        """The state of the form's unit changes only under address_state: replace; 'State: NSW' elsewhere stays, and is
        never made a locality (broad replaces localities)."""
        form = column(
            "Address:",
            "Unit",
            "5",
            "Street No.",
            "12",
            "Street name",
            "Smith Street",
            "Suburb/Town",
            "Penrith",
            "State",
            "NSW",
            "Postcode",
            "2750",
            "",
            "Notes",
            "",
            "State: NSW",
        )
        kept = run(form, profile)
        released_clean(kept)
        assert kept.tokens().count("nsw") == 2
        moved = run(form, profile, {"policy": {"address_state": "replace"}})
        released_clean(moved)
        words = moved.tokens()
        assert words.count("nsw") == 1, "the lone 'State: NSW' stays, the form's state box moved"
        assert not {"penrith", "2750"} & set(words)
