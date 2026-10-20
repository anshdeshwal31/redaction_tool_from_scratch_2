"""A stand-in for the ``mlredact`` CLI, for API tests: same arguments, same fixed output names, same
exit codes (0 released, 2 quarantined, 1 error), same ``error: <CODE> (...)`` line on stderr.  It
loads no models and finishes in milliseconds.

    MLREDACT_API_ENGINE='["python", "tests/api/stub_engine.py"]'

A marker near the start of the input picks the scenario, so each job can ask for its own outcome
(it may follow a ``%PDF-`` header):

    %STUB released [delay=<ms>]   default (also for any input without a marker)
    %STUB quarantined             V4 fails after one re-plan: exit 2, manifest + quarantine.json
    %STUB intake                  rejected as encrypted before analysis: exit 2, no pages
    %STUB error                   exit 1 with ``error: E_CONFIG_INVALID``, nothing written
    %STUB crash                   exit 3 with no reason code, nothing written
    %STUB hang                    never finishes (cancel and timeout tests)

With MLREDACT_STUB_RECORD_ARGS=1 it also writes its argv to <out>/stub-argv.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

W, H = 2481, 3508


def option(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def fail(code: str) -> None:
    sys.stderr.write(f"error: {code} (EnvironmentErrorMl at stub:main:1)\n")
    sys.exit(1)


def pt(r: list[int]) -> list[float]:
    x0, y0, x1, y1 = r
    return [round(v * 0.24, 2) for v in (x0, H - y1, x1, H - y0)]


def box(px: list[int], erased: list[int] | None = None) -> dict[str, Any]:
    b: dict[str, Any] = {"page": 0, "px": px, "pt": pt(px)}
    if erased:
        b["erased_px"] = erased
    return b


def ev(detector: str, rule: str, strength: str = "strong") -> dict[str, Any]:
    return {"detector": detector, "rule": rule, "score": 1, "strength": strength, "version": "1"}


def manifest(
    *, status: str, reasons: list[str], analysed: bool, style: str, profile: str, sha: str, caller: bool
) -> dict[str, Any]:
    """A small manifest that follows schemas/manifest.v1.json (the frontend's contract test checks
    fixtures generated from it)."""

    def replacement(surrogate: str, label: str) -> dict[str, str]:
        return {"surrogate": surrogate} if style == "surrogate" else {"label": label} if style == "label" else {}

    # ``action`` is the policy's decision whatever the style; ``render`` lists the operations drawn.
    drawn = ["blackout"] if style == "blackout" else [style]
    line_drawn = ["blackout"] if style == "blackout" else ["retypeset_region"]
    line = [237, 296, 1240, 367]
    erased_line = [227, 296, 1252, 367]
    mentions = (
        [
            {
                "action": "surrogate",
                "boxes": [box([300, 500, 700, 560])],
                "evidence": [ev("rules.persons", "person.title")],
                "mention_id": "M-000000000000000000000001",
                "render": drawn,
                "token_ids": ["p0000.l0004.w001", "p0000.l0004.w002"],
                "type": "person",
                **replacement("Hayley McDonald", "[PERSON 1]"),
            },
            # Two mentions on one re-typeset line share a single erased box.
            {
                "action": "surrogate",
                "boxes": [box(line, erased_line)],
                "evidence": [ev("rules.address", "address.street")],
                "mention_id": "M-000000000000000000000002",
                "render": line_drawn,
                "token_ids": ["p0000.l0001.w000", "p0000.l0001.w001"],
                "type": "street_address",
                **replacement("217 Hume Street,", "[ADDRESS 1]"),
            },
            {
                "action": "surrogate",
                "boxes": [box(line, erased_line)],
                "evidence": [ev("rules.address", "address.postcode")],
                "mention_id": "M-000000000000000000000003",
                "render": line_drawn,
                "token_ids": ["p0000.l0001.w004"],
                "type": "postcode",
                **replacement("2165", "[POSTCODE 1]"),
            },
            {
                "action": "keep",
                "boxes": [box([300, 700, 600, 760])],
                "evidence": [ev("rules.dates", "date.numeric")],
                "mention_id": "M-000000000000000000000004",
                "render": [],
                "token_ids": ["p0000.l0006.w000"],
                "type": "date",
            },
        ]
        if analysed
        else []
    )
    released = status == "released"
    checks = (
        [
            {"check": c, "metrics": m, "passed": p}
            for c, p, m in [
                ("V1_structure", True, {"pages": 1, "problems": 0}),
                ("V2_text", True, {"hits": 0, "needles": 3}),
                ("V3_pixels", True, {"pages_checked": 1, "pages_failed": 0}),
                ("V4_residual", released, {"replans": 0 if released else 1, "residual_tokens": 0 if released else 2}),
                ("V5_surrogate", True, {"collision": 0, "surrogates": 3}),
                ("V6_integrity", True, {"applied_ops": 3, "planned_ops": 3, "problems": 0}),
            ]
        ]
        if analysed
        else []
    )
    inspection = {
        "annotations_flattened_or_dropped": 0,
        "attachments_dropped": 0,
        "encrypted": False,
        "has_acroform": False,
        "has_javascript": False,
        "has_optional_content": False,
        "has_xfa": False,
        "page_count": 1,
        "user_unit_pages": 0,
    }
    page = {
        "dpi": 300,
        "height_pt": 841.89,
        "height_px": H,
        "index": 0,
        "kind": "image_only",
        "lines": 12,
        "mean_ocr_confidence": 0.98,
        "notice": True,
        "tokens": 60,
        "unexplained_ink": 0.02,
        "width_pt": 595.276,
        "width_px": W,
    }
    region = {
        "action": "remove",
        "box": box([1957, 114, 2184, 341]),
        "detector": "zxing",
        "kind": "barcode",
        "region_id": "G-000000000000000000000001",
        "rule": "barcode.decoded",
        "score": 1,
    }
    residual = {
        "page": 0,
        "pt": pt([300, 900, 500, 950]),
        "px": [300, 900, 500, 950],
        "ref_id": "R-1",
        "type": "person",
    }
    return {
        "detection": {"propagation_rounds": 1 if analysed else 0},
        "input": inspection if analysed else None,
        "job": {
            "input_sha256": sha,
            "page_count": 1 if analysed else 0,
            "profile": profile,
            "render_style": style,
            "source_format": "pdf",
        },
        "low_evidence": (
            [
                {
                    "evidence": [ev("ner.gliner", "ner.span", "weak")],
                    "token_ids": ["p0000.l0009.w003"],
                    "type": "organisation",
                }
            ]
            if analysed
            else []
        ),
        "mentions": mentions,
        "pages": [page] if analysed else [],
        "provenance": {
            "config_hash": "0" * 64,
            "detectors": [{"id": "rules.address", "version": "1"}] if analysed else [],
            "mlredact_version": "0.0.0-stub",
            "models": [],
            "runtime_profile": {"id": "stub-x86_64", "libraries": {}},
            "surrogate": (
                {"key_id": "k-0000000000000000", "scope": "caller" if caller else "document"}
                if style == "surrogate"
                else None
            ),
        },
        "reasons": reasons,
        "regions": [region] if analysed else [],
        "residual_boxes": [residual] if status == "quarantined" and analysed else [],
        "schema_version": "1.1.0",
        "status": status,
        "verification": {"checks": checks, "passed": analysed and released},
    }


def minimal_pdf() -> bytes:
    """One blank A4 page, with a correct cross-reference table."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.276 841.89] >>",
    ]
    body = "%PDF-1.7\n"
    offsets = []
    for i, obj in enumerate(objects):
        offsets.append(len(body))
        body += f"{i + 1} 0 obj\n{obj}\nendobj\n"
    xref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    body += "".join(f"{at:010d} 00000 n \n" for at in offsets)
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return body.encode("ascii")


def main(argv: list[str]) -> None:
    command = argv[0] if argv else ""
    if command == "profiles":
        sys.stdout.write("broad\nclaimant_focused\nclient\nmaximal\n")
        sys.exit(0)
    if command == "selfcheck":
        report = {
            "config_hash": "0" * 64,
            "libraries": {"python": "3.12.0"},
            "mlredact_version": "0.0.0-stub",
            "models_verified": [],
            "profile": option(argv, "--profile") or "broad",
            "resources_verified": 0,
            "runtime_profile": "stub-x86_64",
        }
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        sys.exit(0)
    if command != "run" or len(argv) < 2:
        fail("E_CONFIG_INVALID")
    source = Path(argv[1])
    out_s = option(argv, "--out")
    if not source.is_file() or not out_s:
        fail("E_CONFIG_INVALID")
    out = Path(out_s or "")
    data = source.read_bytes()
    marker = re.search(rb"^%STUB (\w+)(?: delay=(\d+))?", data[:200], re.MULTILINE)
    scenario = marker.group(1).decode() if marker else "released"
    delay_ms = int(marker.group(2)) if marker and marker.group(2) else 0
    style = option(argv, "--style") or "surrogate"
    profile = option(argv, "--profile") or "broad"
    workers = int(option(argv, "--workers") or 1)

    if os.environ.get("MLREDACT_STUB_RECORD_ARGS") == "1":
        out.mkdir(parents=True, exist_ok=True)
        (out / "stub-argv.json").write_text(json.dumps(argv))

    if scenario == "hang":
        while True:
            time.sleep(1)
    if delay_ms:
        time.sleep(delay_ms / 1000)

    def write(m: dict[str, Any], quarantined: bool) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(m, indent=2))
        timings = (
            {
                "analyse_ms": 10,
                "detect_ms": 2,
                "inspect_ms": 1,
                "render_build_ms_0": 3,
                "total_ms": 20,
                "verify_ms_0": 4,
            }
            if m["pages"]
            else {"inspect_ms": 1, "total_ms": 1}
        )
        run = {
            "job_id": "job-" + "0" * 32,
            "mlredact_version": "0.0.0-stub",
            "reasons": m["reasons"],
            "status": m["status"],
            "timings_ms": timings,
            "workers": workers,
        }
        (out / "run.json").write_text(json.dumps(run))
        if quarantined:
            q = {"input_sha256": m["job"]["input_sha256"], "reasons": m["reasons"], "status": "quarantined"}
            (out / "quarantine.json").write_text(json.dumps(q))
        if option(argv, "--sensitive-to"):
            (out / "sensitive-manifest.sealed.json").write_text("{}")

    common = {
        "style": style,
        "profile": profile,
        "sha": hashlib.sha256(data).hexdigest(),
        "caller": option(argv, "--scope-id") is not None,
    }
    if scenario == "error":
        sys.stderr.write("[INFO] third-party log line that must never be kept: Mr John Citizen\n")
        fail("E_CONFIG_INVALID")
    if scenario == "crash":
        sys.stderr.write("Traceback (most recent call last): something about Mr John Citizen\n")
        sys.exit(3)
    if scenario == "intake":
        write(manifest(status="quarantined", reasons=["E_INPUT_ENCRYPTED"], analysed=False, **common), True)
        sys.exit(2)
    if scenario == "quarantined":
        write(manifest(status="quarantined", reasons=["V4_RESIDUAL"], analysed=True, **common), True)
        sys.exit(2)
    write(manifest(status="released", reasons=[], analysed=True, **common), False)
    (out / "redacted.pdf").write_bytes(minimal_pdf())
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
