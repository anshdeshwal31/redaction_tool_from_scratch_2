# mlredact

Local, deterministic PII redaction for Australian medico-legal PDFs (mostly scanned).

The output is a **rebuilt** PDF: every page is re-rendered, PII is removed from the raster, and a new
PDF is constructed from scratch (nothing from the input's object graph survives).  Release is
**fail-closed**: a document is released only if every verification check passes; otherwise it is
quarantined with PII-free reason codes.

## Status

| Phase (see plan) | State |
|---|---|
| P0 Foundations: types, config/profiles, determinism controls, PII-safe logging, model registry | done |
| P1 Safe minimal pipeline: intake, PDFium render, PP-OCRv6 OCR, rule detectors, propagation, PDF rebuild, verification V1-V4/V6, manifest, CLI | done |
| P2 OCR robustness and layout: region detectors (barcodes, faces, signatures, stamps, handwriting, show-through), ink accounting with re-reading, legibility gate, embedded text (engine D), Tesseract (engine B), form geometry (values below / beside labels) | done (engine B awaits validation on the Linux instance; a trained layout model is optional future work) |
| P3 Detection ensemble: OpenAI Privacy Filter (own Viterbi, exact windowing) + GLiNER-PII, contextual allowlist, seeds | done (gazetteers pending) |
| P4 Resolution and surrogates: name parser, clustering, aliases, keyed surrogates, erase + typeset, labels, text layer, notice, V5; claimant-focused roles; Maximal date shift + age top-coding; line re-typesetting (PII-dense blocks, positional hardening) | done (nickname table and née/aka linking pending) |
| P5 Generative tiers: VLM second reader, LLM sweep | planned |
| P6 Productionisation: API, queue, images, observability | planned |

The default render style is `surrogate` (realistic replacements derived from a keyed HMAC).  Styles
`label` (`[PERSON 1]`) and `blackout` are also available.

| profile | what is replaced |
|---|---|
| `broad` (default) | every person, identifier, contact detail, DOB/DoD, street/suburb/postcode, linked organisation; clinical content and other dates kept |
| `claimant_focused` | as broad, but people resolved as professionals (explicit title/post-nominal/role, no subject cue, surname not shared with the claimant's family) and their provider identifiers stay visible |
| `maximal` | as broad, plus every date shifted by one keyed offset per scope (intervals preserved, layout kept) and ages of 90+ top-coded to "90+" |

## Quick start (development)

```bash
uv sync                                  # Python 3.12, locked dependencies
uv run mlredact models fetch             # download + SHA-256-verify pinned models into .models/
uv run mlredact selfcheck                # config, runtime profile, resources, models
export MLREDACT_SURROGATE_KEY=<64+ hex chars from your secret store>
uv run mlredact run input.pdf --out out/ --workers 4 [--scope-id MATTER-123]
uv run mlredact run input.pdf --out out/ --dev-key        # local experiments only: public key
uv run mlredact run input.pdf --out out/ --style blackout # no key needed
uv run mlredact run scan.tiff --out out/ --dev-key        # TIFF / PNG / JPEG scans are accepted too
```

The NER models load once per process (~35 s, ~2.5 GB RAM).  OCR runs in `--workers` processes;
detection runs in the main process with `detection.ner.threads` threads.

Outputs (fixed names — input filenames are never reused because they often contain names):

| file | content |
|---|---|
| `redacted.pdf` | only when released |
| `manifest.json` | deterministic, **no document text**: mentions (type, action, boxes in px and pt, evidence), verification results, provenance |
| `run.json` | operational: job id, timings |
| `quarantine.json` | only when quarantined: reason codes |
| `sensitive-manifest.sealed.json` | only with `--sensitive-to evaluator.pub`: original text, sealed to the evaluator's key |

Exit codes: `0` released, `2` quarantined, `1` usage/environment error.

### Interface for the external evaluator

`manifest.json` follows the JSON Schema in [schemas/manifest.v1.json](schemas/manifest.v1.json)
(strict: unknown fields are rejected, so any change is deliberate; additive changes bump the minor
`schema_version`, anything else the major version and file name).  Per mention it gives the type,
action, evidence (detector, version, rule, strength, score), token ids and boxes in canonical
pixels and PDF points; kept low-evidence candidates are listed separately for error analysis.
Identifiers are derived from positions, so manifests of reruns can be diffed directly.

For error analysis the evaluator can also receive the **sensitive manifest** (each mention's original
text, OCR tokens with alternates, person clusters, low-evidence text).  It is never written in
plaintext: it is sealed (X25519 + HKDF-SHA256 + ChaCha20-Poly1305) to a key the evaluator generates:

```bash
uv run mlredact keys evaluator --out keys/                    # evaluator.key (secret) + evaluator.pub
uv run mlredact run input.pdf --out out/ --sensitive-to keys/evaluator.pub
uv run mlredact sensitive open out/sensitive-manifest.sealed.json --key keys/evaluator.key
```  The schema's
enumerations are tested against the code, and every integration-test manifest is validated against it.

## Tests

```bash
make test               # fast: unit, property, security, verifier mutations, hash-seed determinism
make test-slow          # end-to-end + model-backed (needs `make models`; tens of minutes on CPU)
make determinism        # byte-identical outputs across reruns, worker counts and hash seeds
make leaktest           # no document text in logs, exceptions, manifests or run records
make verify-mutations   # deliberately broken outputs must fail V1-V6
make lint licences      # ruff, mypy --strict, licence gate
```

All fixtures are synthetic (`tools/synthdocs`).  Real documents never enter the repository.

## Layout

```
src/mlredact/
  core/        types, geometry, canonical JSON + content IDs, errors (reason codes only)
  security/    Sensitive[T], allowlisted structured logging, exception sanitising
  config/      pydantic schema, layered loader, built-in profiles (broad, claimant_focused, maximal)
  runtime/     determinism settings, runtime-profile detection
  registry/    pinned model registry (SHA-256), fetch/resolve
  resources/   pinned fonts (Liberation, OFL), surrogate pools, allowlists + SHA256SUMS
  ingest/      sandboxed inspection (pikepdf), canonical rasterisation (PDFium), embedded text, image inputs
  ocr/         PP-OCRv6 engine A, Tesseract engine B, fusion with the PDF text layer (engine D)
  text/        text views with exact offset->token maps, normalisation
  detect/      rule detectors, NER (Privacy Filter, GLiNER), allowlist, fusion, propagation
  resolve/     name parsing, person clustering, roles/gender
  surrogate/   keyed HMAC-DRBG, pools, format-preserving generators, assignment, labels, date shift
  layout/      barcodes (zxing-cpp), faces (YuNet), ink accounting and region classification
  imaging/     ink masks against a local background
  policy/      profile -> actions
  render/      render planning + raster operations
  pdfout/      deterministic encoding + from-scratch PDF builder
  verify/      V1 structure, V2 text leak, V3 pixels, V4 residual re-OCR, V5 surrogates, V6 integrity
  manifest/    non-sensitive manifest; sealed sensitive manifest
  pipeline/    worker pool (spawned processes), shared-memory blobs, job runner
  cli/         Typer CLI
tools/synthdocs/  synthetic test documents
docs/adr/         architecture decision records
```

## Determinism

Outputs are byte-identical for the same input, configuration, models, software and **runtime
profile** (e.g. `linux-x86_64-v3`).  Pin `runtime.expected_profile` in production; workers refuse to
start on a mismatch.  Thread counts are fixed per process, batch sizes are fixed (1), IDs are
derived from positions, and every collection is emitted in canonical order.
