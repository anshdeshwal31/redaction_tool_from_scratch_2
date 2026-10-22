"""``mlredact bench`` in the oracle modes on a synthetic scan, fed the stage files of its own e2e run the way the
benchmark swaps one run's stages into another (its ``materialise_from_run``): the files are already in display points,
and the mentions' labels are mapped to benchmark types through tool 2's published label map.  Every run starts the
way the benchmark starts it: a subprocess from a run folder that is not the backend folder (slow: loads the models).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from synthdocs import SynthDoc, make_ime_letter

from inject_support import PARTIAL, changed_shares
from mlredact.bench.inject import MODE_FILES, InjectedFile
from mlredact.bench.labels import LABEL_MAP
from mlredact.text.normalize import match_key
from mlredact.verify.checks import extract_all_text
from support import EXTRA_CONFIG

pytestmark = [pytest.mark.slow, pytest.mark.models]

MODES = ("oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions")
FILES = (
    "entities.json",
    "manifest.json",
    "mentions.json",
    "output.pdf",
    "plan.json",
    "pseudonyms.json",
    "run.json",
    "run_info.json",
    "text.json",
)
PLANTED = ("person", "phone", "email", "medicare")  # planted values none of which may be read in the output
# The benchmark and its environment, when they sit next to tool 2.
BENCHMARK = Path(__file__).resolve().parents[4] / "redaction_benchmark"
BENCHMARK_PYTHON = next(
    (
        p
        for p in (BENCHMARK / ".venv" / "Scripts" / "python.exe", BENCHMARK / ".venv" / "bin" / "python")
        if p.is_file()
    ),
    None,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _bounds(poly: list[list[float]]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _swapped(e2e: Path, folder: Path, mode: str) -> Path:
    """The e2e run's stage files as ``mode``'s injected inputs (the benchmark's stage swap)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in MODE_FILES[mode]:
        document = _json(e2e / name.value)
        if name is InjectedFile.MENTIONS:
            for m in document["mentions"]:
                m["label"] = LABEL_MAP.get(m["label"], m["label"])
        (folder / name.value).write_text(json.dumps(document), "utf-8")
    return folder


class Runs:
    """``bench`` on the scan in e2e, then in each oracle mode on the e2e run's files: each run once, on first use."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._done: dict[str, subprocess.CompletedProcess[str]] = {}

    def out(self, mode: str) -> Path:
        """The run's output folder (the run exited 0)."""
        if mode not in self._done:
            inject = "" if mode == "e2e" else str(_swapped(self.out("e2e"), self.folder / f"inject-{mode}", mode))
            self._done[mode] = self._bench(mode, inject)
        proc = self._done[mode]
        assert proc.returncode == 0, proc.stderr[-2000:]
        return self.folder / mode

    def _bench(self, mode: str, inject_dir: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        if EXTRA_CONFIG:  # e.g. the laptop arrangement without NER and Tesseract
            env["MLREDACT_CONFIG"] = os.pathsep.join(str(p.resolve()) for p in EXTRA_CONFIG)
        args = ["--input", str(self.folder / "input.pdf"), "--workdir", str(self.folder / mode)]
        args += ["--mode", mode, "--inject-dir", inject_dir]
        return subprocess.run(
            [sys.executable, "-m", "mlredact.cli.main", "bench", *args, "--scope-id", "", "--seed", "0"],
            cwd=self.folder,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )


@pytest.fixture(scope="module")
def scan() -> SynthDoc:
    return make_ime_letter(scanned=True, seed=7, artefacts=True)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, scan: SynthDoc) -> Runs:
    folder = tmp_path_factory.mktemp("run")
    (folder / "input.pdf").write_bytes(scan.pdf)
    return Runs(folder)


# ----------------------------------------------------------------------------------------- every mode
@pytest.mark.parametrize("mode", MODES)
def test_every_file_is_handed_over_and_names_only_what_the_others_list(runs: Runs, mode: str) -> None:
    out = runs.out(mode)
    info = _json(out / "run_info.json")
    held = [] if info["release"]["released"] else ["quarantine.json"]
    assert sorted(p.name for p in out.iterdir()) == sorted([*FILES, *held])
    assert {k: info["config"][k] for k in ("mode", "injected")} == {
        "mode": mode,
        "injected": sorted(f.kind for f in MODE_FILES[mode]),
    }
    tokens = {t["id"] for p in _json(out / "text.json")["pages"] for t in p["tokens"]}
    mentions = _json(out / "mentions.json")["mentions"]
    ids = {m["id"] for m in mentions}
    assert len(ids) == len(mentions) and all(set(m.get("tokens", [])) <= tokens for m in mentions)
    plan = _json(out / "plan.json")["regions"]
    assert len({r["id"] for r in plan}) == len(plan)
    named = {i for r in plan for i in r["mentions"]}
    named |= {i for c in _json(out / "entities.json")["clusters"] for i in c["mentions"]}
    named |= {m["mention"] for m in _json(out / "pseudonyms.json")["mentions"]}
    assert named <= ids


@pytest.mark.parametrize("mode", MODES)
def test_polygons_are_points_of_the_displayed_page(runs: Runs, mode: str) -> None:
    out = runs.out(mode)
    (page,) = _json(out / "manifest.json")["pages"]
    width, height = page["width_pt"], page["height_pt"]
    polys = [t["poly"] for t in _json(out / "text.json")["pages"][0]["tokens"]]
    polys += [r["poly"] for m in _json(out / "mentions.json")["mentions"] for r in m["regions"]]
    polys += [r["poly"] for r in _json(out / "plan.json")["regions"]]
    for poly in polys:
        x0, y0, x1, y1 = _bounds(poly)
        assert -0.01 <= x0 <= x1 <= width + 0.01 and -0.01 <= y0 <= y1 <= height + 0.01


@pytest.mark.parametrize("mode", ("e2e", *MODES))
def test_no_planted_identifier_can_be_read_in_the_output(runs: Runs, scan: SynthDoc, mode: str) -> None:
    text = match_key(extract_all_text((runs.out(mode) / "output.pdf").read_bytes()))
    assert [p.kind for p in scan.planted if p.kind in PLANTED and match_key(p.text) in text] == []


@pytest.mark.parametrize("mode", ("e2e", *MODES))
def test_the_benchmark_accepts_every_file(runs: Runs, mode: str) -> None:
    """``rb candidate check`` with tool 2's candidate manifest (it reads the files, and writes nothing)."""
    if BENCHMARK_PYTHON is None:
        pytest.skip("the benchmark and its environment are not next to tool 2")
    manifest = BENCHMARK / "candidates" / "mlredact-client.yaml"
    proc = subprocess.run(
        [
            str(BENCHMARK_PYTHON),
            "-m",
            "benchmark.cli",
            "candidate",
            "check",
            str(runs.out(mode)),
            "--manifest",
            str(manifest),
        ],
        cwd=BENCHMARK,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0 and proc.stdout.split()[-1:] == ["valid"], proc.stdout + proc.stderr[-2000:]


# ------------------------------------------------------------------------------------------ each mode
def test_oracle_text_reads_the_injected_text_and_nothing_else(runs: Runs) -> None:
    """The words, their boxes, confidences and order are e2e's; so are its lines, but where a line's words lie more
    than 3 word heights apart (the notes count those splits); tool 2 numbers the words by their place."""
    e2e, oracle = _json(runs.out("e2e") / "text.json"), _json(runs.out("oracle_text") / "text.json")
    assert oracle["source"] == e2e["source"]

    def words(text: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [(p["index"], t["text"], t["poly"], t["conf"], t["order"]) for p in text["pages"] for t in p["tokens"]]

    def breaks(text: dict[str, Any]) -> list[bool]:
        return [
            a["line"] != b["line"] for p in text["pages"] for a, b in zip(p["tokens"], p["tokens"][1:], strict=False)
        ]

    assert words(oracle) == words(e2e)
    notes = _json(runs.out("oracle_text") / "run_info.json")["notes"]
    assert "(0 empty, 0 with a repeated id, 0 beyond the last page, 0 page entries merged)" in notes
    found = re.search(r"\((\d+) split at a wide gap\)", notes)
    assert found is not None
    assert all(o for e, o in zip(breaks(e2e), breaks(oracle), strict=True) if e)
    assert sum(breaks(oracle)) - sum(breaks(e2e)) == int(found.group(1))


def test_oracle_mentions_redacts_what_e2e_redacted(runs: Runs) -> None:
    """Every mention e2e acted on, injected back, is reported under its id with the same words and a label of the same
    benchmark type, and every fake e2e drew is drawn again (but for a mention whose words another one holds too: it
    shows that one's fake).  Known differences, not checked here: from oracle_mentions on tool 2 finds no region
    itself (show-through and graphics are left as they are), and verification re-plans only removed values."""
    e2e, oracle = runs.out("e2e"), runs.out("oracle_mentions")
    texts = {
        run: {t["id"]: t["text"] for p in _json(run / "text.json")["pages"] for t in p["tokens"]}
        for run in (e2e, oracle)
    }
    injected = {m["id"]: m for m in _json(e2e / "mentions.json")["mentions"]}
    reported = {m["id"]: m for m in _json(oracle / "mentions.json")["mentions"]}
    assert set(reported) == set(injected)
    for mention_id, m in injected.items():
        assert LABEL_MAP[reported[mention_id]["label"]] == LABEL_MAP[m["label"]], mention_id
        if m.get("tokens"):
            words = [texts[oracle][t] for t in reported[mention_id]["tokens"]]
            assert words == [texts[e2e][t] for t in m["tokens"]], mention_id
    # A region e2e found becomes a text mention where words lie inside it (a stamp's): it may share them.
    held = {i: set(m.get("tokens", ())) for i, m in reported.items()}
    shared = {i for i in held for j in held if i != j and held[i] & held[j]}
    drawn = {p["mention"]: p["replacement"] for p in _json(e2e / "pseudonyms.json")["mentions"]}
    again = {p["mention"]: p["replacement"] for p in _json(oracle / "pseudonyms.json")["mentions"]}
    assert drawn and {i: r for i, r in drawn.items() if i not in shared}.items() <= again.items()


def test_oracle_entities_takes_the_injected_clusters(runs: Runs) -> None:
    injected = {c["id"]: sorted(c["mentions"]) for c in _json(runs.out("e2e") / "entities.json")["clusters"]}
    out = runs.out("oracle_entities")
    assert {c["id"]: sorted(c["mentions"]) for c in _json(out / "entities.json")["clusters"]} == injected
    assert {e["cluster"] for e in _json(out / "pseudonyms.json")["entries"]} <= {*injected, None}


def test_oracle_regions_applies_every_region_of_the_injected_plan(runs: Runs, scan: SynthDoc) -> None:
    """Each region e2e's plan.json lists is applied, and changes the page as the benchmark measures it (at least
    "partial")."""
    injected = _json(runs.out("e2e") / "plan.json")["regions"]
    out = runs.out("oracle_regions")
    plan = _json(out / "plan.json")["regions"]
    assert sorted(r["id"] for r in plan) == sorted(r["id"] for r in injected)
    shares = changed_shares(scan.pdf, (out / "output.pdf").read_bytes(), plan)
    assert {i: share for i, share in sorted(shares.items()) if share < PARTIAL} == {}
