"""Score redaction runs of the synthetic corpus against its answer keys and write a readable report.

    PYTHONPATH=tools uv run python -m synthdocs.score --corpus out/synth_corpus --runs out/synth_runs

``--runs`` holds one folder per document, named after its slug, as ``mlredact run --out`` writes it
(``redacted.pdf`` or ``quarantine.json``, plus ``manifest.json``; ``seconds.txt`` if the run was timed).

Checked per document:
* leaks - each value the client profile replaces is looked for in the redacted pages, read back with
  OCR (PP-OCR, and Tesseract when installed); a value either reader still finds is a leak.  Values
  neither reader finds in the *original* cannot be measured and are listed separately.
* damage - each value that must stay (dates, suburbs, medicines, ...) must still be readable.
* hidden text - replaced values must not survive in the output's text layer (copy and paste).
* picture areas - each signature, stamp, handwriting, barcode, logo, photo or show-through must
  overlap a region the tool's manifest says it handled.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlredact.config.loader import load_config
from mlredact.ingest.render import open_document, render_page
from mlredact.ocr.ppocr import PPOCREngine
from mlredact.registry.models import models_dir
from mlredact.verify.checks import extract_all_text

PORTRAIT = (595.276, 841.89)
TOLERANCE_PT = 18.0  # pages are tilted up to 2.5 degrees, which moves content up to ~18 pt
REMOVED_WITH_REGION = {"stamp", "handwriting"}  # values inside areas the policy removes whole


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class _Text:
    tokens: list[str]
    compact: str

    @classmethod
    def of(cls, text: str) -> _Text:
        toks = _tokens(text)
        return cls(toks, "".join(toks))

    def count(self, needle: str) -> int:
        """Occurrences of ``needle``; long values ignore spacing (OCR splits and joins digit groups)."""
        seq = _tokens(needle)
        compact = "".join(seq)
        if len(compact) >= 6:
            return self.compact.count(compact)
        n = len(seq)
        return sum(self.tokens[i : i + n] == seq for i in range(len(self.tokens) - n + 1))


def _measurable(text: str) -> bool:
    c = "".join(_tokens(text))
    return len(c) >= 2 and not (c.isdigit() and len(c) < 4)


class Readers:
    """The OCR engines used to read pages back: PP-OCR always, Tesseract when it is installed."""

    def __init__(self) -> None:
        self.cfg = load_config("client")
        mdir = models_dir(self.cfg.runtime.models_dir)
        self.ppocr = PPOCREngine(self.cfg.ocr.engine_a, self.cfg.runtime.threads, mdir)
        self.tesseract: Any = None
        try:
            from mlredact.ocr.tesseract import TesseractEngine, binary_version

            if binary_version() is not None:
                self.tesseract = TesseractEngine(self.cfg.ocr.engine_b, mdir)
        except Exception:  # no Tesseract here: PP-OCR alone
            self.tesseract = None

    @property
    def names(self) -> list[str]:
        return ["PP-OCR", *(["Tesseract"] if self.tesseract else [])]

    def read(self, pdf: bytes) -> dict[str, _Text]:
        lines: dict[str, list[str]] = {n: [] for n in self.names}
        doc = open_document(pdf, self.cfg.render)
        try:
            for i in range(len(doc)):
                page = render_page(doc, i, self.cfg.render, self.cfg.limits)
                lines["PP-OCR"].extend(line.text for line in self.ppocr.ocr_page(page.rgb, i))
                if self.tesseract:
                    lines["Tesseract"].extend(w.text for w in self.tesseract.words(page.rgb, page.geometry.dpi))
        finally:
            doc.close()
        return {n: _Text.of("\n".join(t)) for n, t in lines.items()}


# -- values ----------------------------------------------------------------------------------------


def _score_values(key: dict, orig: dict[str, _Text], out: dict[str, _Text], layer: _Text) -> list[dict]:
    keeps = [v for v in key["values"] if v["action"] == "keep"]
    results = []
    for v in key["values"]:
        rec = {k: v[k] for k in ("kind", "text", "action", "count", "pages", "where")}
        if not _measurable(v["text"]):
            rec["status"] = "too short to check"
            results.append(rec)
            continue
        n_orig = max(t.count(v["text"]) for t in orig.values())
        out_by = {name: t.count(v["text"]) for name, t in out.items()}
        n_out = max(out_by.values())
        if v["action"] == "keep":
            if set(v["where"]) <= REMOVED_WITH_REGION:
                rec["status"] = "inside an area removed whole"
            elif n_orig == 0:
                rec["status"] = "not readable in the original"
            elif n_out >= min(n_orig, v["count"]):
                rec["status"] = "kept"
            else:
                rec["status"] = "partly removed" if n_out else "removed"
        else:
            # Occurrences that belong to values which must stay ('hope' the word, not Ms Hope).
            allowed = sum(k["count"] * _Text.of(k["text"]).count(v["text"]) for k in keeps)
            if n_orig <= allowed:
                rec["status"] = "not readable in the original"
            elif n_out > allowed:
                rec["status"] = "leaked"
                rec["seen_by"] = [name for name, c in out_by.items() if c > allowed]
            else:
                rec["status"] = "replaced"
            rec["in_text_layer"] = layer.count(v["text"]) > allowed
        results.append(rec)
    return results


# -- picture areas -------------------------------------------------------------------------------------


Box = tuple[float, float, float, float]


def _scanned_box(box: Box, layout: dict) -> Box:
    """A box on the typeset page, moved to where the scan put it (pt, origin top-left)."""
    x0, y0, x1, y1 = box
    w, h = PORTRAIT
    if layout["landscape"]:  # turned a quarter counter-clockwise: (x, y) -> (y, page width - x)
        x0, y0, x1, y1 = y0, h - x1, y1, h - x0
    if layout["upside_down"]:
        x0, y0, x1, y1 = w - x1, h - y1, w - x0, h - y0
    return x0, y0, x1, y1


def _manifest_box(box: dict, page_h: float) -> Box:
    x0, y0, x1, y1 = box["pt"]
    return x0, page_h - y1, x1, page_h - y0


def _overlaps(a: Box, b: Box, pad: float) -> bool:
    return a[0] - pad < b[2] and b[0] < a[2] + pad and a[1] - pad < b[3] and b[1] < a[3] + pad


def _score_regions(key: dict, manifest: dict) -> list[dict]:
    heights = {p["index"]: p["height_pt"] for p in manifest["pages"]}
    tool_regions = [
        (r["kind"], r["action"], r["box"]["page"], _manifest_box(r["box"], heights[r["box"]["page"]]))
        for r in manifest.get("regions", [])
    ]
    text_boxes = [
        (b["page"], _manifest_box(b, heights[b["page"]]))
        for m in manifest.get("mentions", [])
        for b in m.get("boxes", [])
    ] + [(b["page"], _manifest_box(b, heights[b["page"]])) for b in manifest.get("residual_boxes", [])]
    results = []
    for r in key["regions"]:
        page = r["page"] - 1
        rec = {"page": r["page"], "kind": r["kind"], "expected": r["action"], "note": r["note"]}
        if r["kind"] == "faint_ink":
            hits = [t for t in tool_regions if t[2] == page and t[0] == "faint_ink"]
        else:
            box = _scanned_box(tuple(r["box_pt"]), key["page_layout"][page])  # type: ignore[arg-type]
            hits = [t for t in tool_regions if t[2] == page and _overlaps(box, t[3], TOLERANCE_PT)]
        handled = [t for t in hits if t[1] != "keep"]
        if handled:
            rec["status"] = "handled"
            rec["by_tool_as"] = sorted({f"{k} ({a})" for k, a, _, _ in handled})
        elif hits:
            rec["status"] = "found but kept"
            rec["by_tool_as"] = sorted({f"{k} ({a})" for k, a, _, _ in hits})
        elif r["kind"] != "faint_ink" and any(p == page and _overlaps(box, b, 0) for p, b in text_boxes):
            rec["status"] = "only the text inside was replaced"
        else:
            rec["status"] = "missed"
        results.append(rec)
    return results


# -- documents ---------------------------------------------------------------------------------------


def score_document(key: dict, original: bytes, run_dir: Path, readers: Readers) -> dict:
    res: dict[str, Any] = {k: key[k] for k in ("document", "tier", "title", "pages", "challenges")}
    res["slug"] = run_dir.name
    seconds = run_dir / "seconds.txt"
    res["seconds"] = float(seconds.read_text().strip()) if seconds.exists() else None
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else None
    out_pdf = run_dir / "redacted.pdf"
    if not out_pdf.exists():
        quarantine = run_dir / "quarantine.json"
        res["status"] = "quarantined" if quarantine.exists() else "no output"
        res["reasons"] = json.loads(quarantine.read_text("utf-8"))["reasons"] if quarantine.exists() else []
        return res
    res["status"] = "released"
    data = out_pdf.read_bytes()
    res["values"] = _score_values(key, readers.read(original), readers.read(data), _Text.of(extract_all_text(data)))
    res["regions"] = _score_regions(key, manifest) if manifest else []
    if manifest:
        res["tool"] = {
            "mentions_by_type": dict(Counter(m["type"] for m in manifest.get("mentions", []))),
            "regions_by_kind": dict(Counter(r["kind"] for r in manifest.get("regions", []))),
            "low_evidence_kept": len(manifest.get("low_evidence", [])),
            "verification_passed": manifest.get("verification", {}).get("passed"),
            "mean_ocr_confidence": [p.get("mean_ocr_confidence") for p in manifest.get("pages", [])],
        }
    return res


def _tally(res: dict) -> dict[str, int]:
    values = res.get("values", [])
    rep = [v for v in values if v["action"] != "keep"]
    keep = [v for v in values if v["action"] == "keep"]
    return {
        "replace_measured": sum(v["status"] in ("replaced", "leaked") for v in rep),
        "replaced": sum(v["status"] == "replaced" for v in rep),
        "leaked": sum(v["status"] == "leaked" for v in rep),
        "keep_measured": sum(v["status"] in ("kept", "removed", "partly removed") for v in keep),
        "kept": sum(v["status"] == "kept" for v in keep),
        "damaged": sum(v["status"] in ("removed", "partly removed") for v in keep),
        "text_layer_leaks": sum(bool(v.get("in_text_layer")) for v in rep),
        "regions": len(res.get("regions", [])),
        "regions_handled": sum(r["status"] == "handled" for r in res.get("regions", [])),
        "unmeasurable": sum(v["status"] in ("not readable in the original", "too short to check") for v in values),
    }


# -- report -------------------------------------------------------------------------------------------


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _plain_summary(res: dict, t: dict[str, int]) -> str:
    if res["status"] != "released":
        why = ", ".join(res.get("reasons", [])) or "no reason recorded"
        return (
            f"The tool did not release this document: it held it back ({res['status']}) because its own checks "
            f"failed ({why}). Nothing can be scored for it."
        )
    parts = [
        f"The tool replaced {t['replaced']} of the {t['replace_measured']} personal details that could be checked."
    ]
    if t["leaked"]:
        leaked = [v for v in res["values"] if v["status"] == "leaked"]
        shown = ", ".join(f"'{v['text']}' ({v['kind']})" for v in leaked[:5])
        more = f" and {len(leaked) - 5} more" if len(leaked) > 5 else ""
        parts.append(f"{t['leaked']} can still be read in the redacted copy: {shown}{more}.")
    else:
        parts.append("None of them can still be read in the redacted copy.")
    if t["damaged"]:
        damaged = [v for v in res["values"] if v["status"] in ("removed", "partly removed")]
        shown = ", ".join(f"'{v['text']}' ({v['kind']})" for v in damaged[:5])
        more = f" and {len(damaged) - 5} more" if len(damaged) > 5 else ""
        parts.append(f"It also removed {t['damaged']} detail(s) that should have stayed: {shown}{more}.")
    else:
        parts.append(f"All {t['keep_measured']} details that should stay (dates, places, medicines, ...) are intact.")
    if t["regions"]:
        missed = [r for r in res["regions"] if r["status"] != "handled"]
        parts.append(f"It dealt with {t['regions_handled']} of the {t['regions']} picture areas")
        if missed:
            parts[-1] += "; not handled: " + ", ".join(f"{r['kind']} on page {r['page']}" for r in missed) + "."
        else:
            parts[-1] += "."
    if t["text_layer_leaks"]:
        parts.append(f"{t['text_layer_leaks']} replaced value(s) can still be copied out of the hidden text layer.")
    return " ".join(parts)


def report_markdown(results: list[dict], readers: list[str]) -> str:
    out = [
        "# Redaction test report - synthetic scanned documents",
        "",
        f"Read back with: {', '.join(readers)}. Policy: client profile.",
        "",
        "## How to read this",
        "",
        "- **Replaced** - a personal detail the policy replaces, no longer readable on the redacted pages.",
        "- **Leaked** - still readable after redaction. This is the number that matters most.",
        "- **Kept / damaged** - details that must stay (dates, suburbs, postcodes, ages, medicines, test names, "
        "citations); 'damaged' means the tool removed or changed one of them.",
        "- **Picture areas** - signatures, stamps, handwriting, barcodes, logos, photos and text showing through "
        "from the back of the sheet; 'handled' means the tool's own record shows it treated that area.",
        "- **Not measurable** - a value the readers could not read even in the original scan (or one too short to "
        "search for, like a one-digit age). These are left out of the percentages.",
        "",
        "## Summary",
        "",
        "| Document | Tier | Status | Time | Replaced | Leaked | Kept | Damaged | Picture areas | Hidden-text leaks |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for res in results:
        t = _tally(res)
        secs = f"{res['seconds']:.0f} s" if res.get("seconds") else "-"
        replaced = _pct(t["replaced"], t["replace_measured"])
        out.append(
            f"| {res['slug']} | {res['tier']} | {res['status']} | {secs} | {replaced} "
            f"| {t['leaked']} | {_pct(t['kept'], t['keep_measured'])} | {t['damaged']} "
            f"| {_pct(t['regions_handled'], t['regions'])} | {t['text_layer_leaks']} |"
        )
    for res in results:
        t = _tally(res)
        out += ["", f"## {res['slug']} - {res['title']} ({res['tier']}, {res['pages']} pages)", ""]
        out += [_plain_summary(res, t), "", "What this document tests:", ""]
        out += [f"- {c}" for c in res["challenges"]]
        if res["status"] != "released":
            continue
        values = res["values"]
        leaked = [v for v in values if v["status"] == "leaked"]
        if leaked:
            out += [
                "",
                "### Still readable after redaction",
                "",
                "| Value | Kind | Pages | Where | Seen by | Also in hidden text |",
            ]
            out += ["|---|---|---|---|---|---|"]
            out += [
                f"| {_cell(v['text'])} | {v['kind']} | {', '.join(map(str, v['pages']))} | {', '.join(v['where'])} "
                f"| {', '.join(v['seen_by'])} | {'yes' if v.get('in_text_layer') else 'no'} |"
                for v in leaked
            ]
        damaged = [v for v in values if v["status"] in ("removed", "partly removed")]
        if damaged:
            out += [
                "",
                "### Removed but should have stayed",
                "",
                "| Value | Kind | Pages | What happened |",
                "|---|---|---|---|",
            ]
            out += [
                f"| {_cell(v['text'])} | {v['kind']} | {', '.join(map(str, v['pages']))} | {v['status']} |"
                for v in damaged
            ]
        if res["regions"]:
            out += ["", "### Picture areas", "", "| Page | Kind | Policy says | Result | Tool recorded it as |"]
            out += ["|---|---|---|---|---|"]
            out += [
                f"| {r['page']} | {r['kind']} | {r['expected']} | {r['status']} | "
                f"{', '.join(r.get('by_tool_as', [])) or '-'} |"
                for r in res["regions"]
            ]
        unmeasured = [v for v in values if v["status"] in ("not readable in the original", "too short to check")]
        if unmeasured:
            out += ["", "### Not measurable", ""]
            out += [f"- '{v['text']}' ({v['kind']}): {v['status']}" for v in unmeasured]
        replaced = [v for v in values if v["status"] == "replaced"]
        out += ["", f"<details><summary>Replaced correctly ({len(replaced)})</summary>", ""]
        out += [f"- {v['text']} ({v['kind']})" for v in replaced]
        out += ["", "</details>"]
        if res.get("tool"):
            tool = res["tool"]
            out += ["", "### What the tool reported", ""]
            out += [
                f"- Its own final checks passed: {tool['verification_passed']}",
                "- Details found, by type: "
                f"{', '.join(f'{k} {v}' for k, v in sorted(tool['mentions_by_type'].items()))}",
                f"- Picture areas found, by kind: "
                f"{', '.join(f'{k} {v}' for k, v in sorted(tool['regions_by_kind'].items())) or 'none'}",
                f"- Candidates it saw but left alone (too little evidence): {tool['low_evidence_kept']}",
                f"- Mean OCR confidence per page: "
                f"{', '.join(f'{c:.2f}' if isinstance(c, float) else '-' for c in tool['mean_ocr_confidence'])}",
            ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=Path("out/synth_corpus"))
    ap.add_argument("--runs", type=Path, default=Path("out/synth_runs"))
    ap.add_argument("--out", type=Path, help="report folder (default: <runs>/report)")
    args = ap.parse_args(argv)
    out_dir = args.out or args.runs / "report"
    readers = Readers()
    results = []
    for run_dir in sorted(p for p in args.runs.iterdir() if p.is_dir() and p != out_dir):
        key_path = args.corpus / f"{run_dir.name}.answers.json"
        if not key_path.exists():
            continue
        key = json.loads(key_path.read_text("utf-8"))
        original = (args.corpus / key["document"]).read_bytes()
        results.append(score_document(key, original, run_dir, readers))
        print(f"scored {run_dir.name}")  # noqa: T201 - CLI progress
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", "utf-8")
    (out_dir / "report.md").write_text(report_markdown(results, readers.names), "utf-8")
    print(f"report: {out_dir / 'report.md'}")  # noqa: T201 - CLI progress


if __name__ == "__main__":
    main()
