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
| `client` (default) | the client's policy decisions: every person (professionals included), organisation, contact detail, identifier, file path or file name and street line; every date (DOB/DoD included), age, suburb/town/place of birth and postcode kept; signatures, faces and photos replaced by synthetic stand-ins (pen strokes, a silhouette); figures and body charts kept |
| `broad` | every person, identifier, contact detail, file path, DOB/DoD, street/suburb/postcode, linked organisation; clinical content and other dates kept |
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
| `held-back.NOT-RELEASED.pdf` | only with `--keep-held-back`, when quarantined after the output was built: the redacted copy that failed verification, for testing and review. Never treat it as a release |

Exit codes: `0` released, `2` quarantined, `1` usage/environment error.

### HTTP API (for the frontend)

`mlredact serve` runs a small FastAPI service that the web frontend ([../frontend](../frontend))
calls. It runs the CLI above once per job, one job at a time, and keeps the same rules: no document
text, no input filenames and no scope ids are stored, and inputs are deleted as soon as a job ends.

The one exception is a job created with `keepOriginal: true`, for the frontend's side-by-side view.
When such a job is released or quarantined after its pages were analysed, its document moves to
`<jobs>/<id>/original/original.pdf` instead of being deleted. A scan is stored as the same image-only
PDF the engine builds from it, converted in a separate process. The original is served like an
artefact and deleted when the job is deleted or when `MLREDACT_API_ORIGINAL_RETENTION_DAYS` have
passed. Every signed-in user can open it.

```bash
export MLREDACT_API_TOKEN=<32+ random characters, shared with the frontend's server>
export MLREDACT_API_CORS_ORIGINS=http://127.0.0.1:3100,https://<your-vercel-app>.vercel.app
export MLREDACT_SURROGATE_KEY=<64+ hex chars>     # or MLREDACT_API_ALLOW_DEV_KEY=1 for synthetic tests
uv run mlredact serve                              # http://127.0.0.1:8100
```

Two kinds of caller:

- **The frontend's server**, with `Authorization: Bearer <MLREDACT_API_TOKEN>`. It creates jobs, reads
  status, cancels, deletes and runs the self-check.
- **A signed-in user's browser**, with a signed link from a job response. It uploads the document and
  downloads artefacts directly, so files never pass through the frontend's host. Each link is tied to
  one job and one file, and expires after minutes. CORS is allowed only for `MLREDACT_API_CORS_ORIGINS`.

| Route | Who | What |
|---|---|---|
| `GET /v1/health` | anyone | liveness |
| `POST /v1/jobs` | token | create a job from its options; returns a signed upload path |
| `PUT /v1/jobs/{id}/input?token=` | signed link | the document's bytes |
| `GET /v1/jobs`, `GET /v1/jobs/{id}` | token | list; status, summary, run record and signed artefact links |
| `GET /v1/jobs/{id}/files/{name}?token=` | signed link | an artefact or the kept `original.pdf` (`&download=true` for an attachment) |
| `POST /v1/jobs/{id}/cancel`, `DELETE /v1/jobs/{id}` | token | stop, or remove with its artefacts |
| `GET /v1/system`, `POST /v1/system/selfcheck` | token | configuration; `mlredact selfcheck` |

Errors are `{"error": "<CODE>"}` only. Request validation errors never echo the request back, because
it may hold seed values.

| Setting | Default | Purpose |
|---|---|---|
| `MLREDACT_API_TOKEN` | required | shared secret, at least 32 characters; also keys the signed links |
| `MLREDACT_API_CORS_ORIGINS` | none | browser origins allowed to use signed links (the frontend's addresses) |
| `MLREDACT_API_CONFIG` | none | deployment YAMLs passed to every run as `--config` (path-separator separated) |
| `MLREDACT_API_JOBS_DIR` | `.jobs` | job records and artefacts |
| `MLREDACT_API_WORKERS` | `2` | default worker processes per job |
| `MLREDACT_API_ALLOW_DEV_KEY` | off | `1` lets jobs use the public development surrogate key |
| `MLREDACT_API_MAX_UPLOAD_MB` | `200` | upload limit |
| `MLREDACT_API_JOB_TIMEOUT_S` | `10800` | wall-clock limit per job |
| `MLREDACT_API_UPLOAD_TTL_S`, `MLREDACT_API_LINK_TTL_S` | `900` | lifetime of upload and download links |
| `MLREDACT_API_ORIGINAL_RETENTION_DAYS` | `7` | how long a kept original stays (0 to 365); `0` refuses `keepOriginal` |
| `MLREDACT_API_ENGINE` | this Python's CLI | engine command as a JSON array (tests point it at a stub) |

**Exposing it from EC2.** `serve` binds `127.0.0.1` by default. Pick one of these:

- **Frontend on your laptop:** use an SSH tunnel, `ssh -L 8100:127.0.0.1:8100 ubuntu@<ec2>`. The
  frontend then uses `http://127.0.0.1:8100`. Traffic is encrypted inside SSH and no new ports open.
- **Frontend on Vercel:** the API needs an `https://` address. With a domain, put a TLS proxy (for
  example Caddy) in front and open ports 443 and 80 (80 only for the certificate). Without one, for
  testing only, run `cloudflared tunnel --url http://127.0.0.1:8100`; it prints a temporary https
  address, and documents pass through Cloudflare.

The API is tested in `tests/api/` against a stub engine (`tests/api/stub_engine.py`). The stub takes
the same arguments and exit codes as the CLI, loads no models, and the first line of its input picks
the outcome.

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

### Benchmark adapter

`mlredact bench` connects mlredact to the redaction benchmark (`redaction_benchmark`, candidate manifest
`candidates/mlredact-client.yaml`).  It is for the evaluator only, never for production: the benchmark
runs it once per document and grades what it writes.

```bash
uv run mlredact bench --input input.pdf --workdir out/ [--scope-id MATTER-123] [--seed 0] [--config x.yaml]
uv run mlredact bench --mode oracle_mentions --inject-dir inject/ --input input.pdf --workdir out/
uv run mlredact bench-version            # 0.1.0+g<commit>, with .dirty when backend/ has uncommitted changes
```

It runs the normal pipeline (default profile `client`) with a trace and writes, into `--workdir` (the stage files
as `e2e` and `oracle_text` write them; the other oracle modes report the injected mentions, see **Modes**):

| file | content |
|---|---|
| `output.pdf` | the redacted PDF, **also when the tool does not release it**: held back by its verification, or built before a later step failed (the notes in run_info.json say which) |
| `run_info.json` | version, configuration, model ids, seed, times, notes, and `release`: the tool's own verdict (`released`, `checks_failed` reason codes) |
| `text.json` | every token: text, box, confidence, line, reading order |
| `mentions.json` | what the tool decided to replace or remove: text mentions (with tokens and token boxes) and PII regions (signature, face, photo, handwriting, barcode, logo, stamp, illegible).  Values the policy keeps (dates, places under `client`) are left out (from `oracle_mentions` on: every injected mention) |
| `entities.json` | one cluster per resolved person |
| `plan.json` | one region per final render operation, with the box it actually erased (`replace` or `remove`; faint-ink suppression is not listed; in `oracle_regions`: the injected plan regions) |
| `pseudonyms.json` | every replacement text drawn, per mention and as original -> replacement (a surrogate whose mention was blacked out instead, e.g. on a sideways line, is not listed) |
| `manifest.json`, `run.json`, `quarantine.json` | as for `run` |

Unlike the manifest, these stage files **contain document text** (the benchmark scores OCR and detection
against its ground truth): they belong only in the benchmark's git-ignored run folder.  No document text
is printed or logged.  Boxes are in `display_pt` (points of the page as displayed, origin top-left): the canonical
raster is the displayed page, so a pixel box converts with its page's own dpi, pt = px * 72 / dpi.
`run` and the HTTP API are unchanged and stay fail-closed.

**Modes.**  `e2e` (the default) runs every stage.  The oracle modes take perfect upstream input from the
benchmark (its ground truth, or another tool's stage files) in `--inject-dir`, in the stage-file formats above
and in `display_pt`, and use it instead of tool 2's own stages.  Each mode injects one file more than the one
before it:

| mode | reads | instead of tool 2's | still tool 2's own |
|---|---|---|---|
| `oracle_text` | `text.json` | OCR: the injected tokens, in their reading order and lines, are the text | region detection (signatures, stamps, barcodes, ...) around that text, detection, the policy (with its keeps), name clustering, surrogates, planning |
| `oracle_mentions` | + `mentions.json` | region detection, detection and the policy: exactly the injected mentions are redacted | name parsing and clustering, surrogates, planning |
| `oracle_entities` | + `entities.json` | person clustering: the injected clusters decide the given-name gender vote and how initials are filled; surname and given-name fakes stay keyed by the name as written (as in `e2e`) | name parsing, surrogates, planning |
| `oracle_regions` | + `plan.json` | planning: exactly the injected regions are applied, without padding | the decisions and surrogates (what is drawn in each region), typesetting, stand-ins |

Rendering, the PDF build and checks V1-V6 run in every mode and give the verdict as in `e2e`.  In every oracle
mode run_info.json records `mode`, `injected` and `ner_enabled`.  A `dpi` given with the injected files is not
used: they are in display points.

- From `oracle_mentions` on every injected mention is acted on: where the profile would keep its type (dates
  and places under `client`) it gets a surrogate.  A mention becomes tool 2's mentions over whole words; a
  signature, handwriting or an image, and an anchor its words do not explain (a logo around a name), become
  regions, which the profile's region actions replace with a stand-in or remove.  A profile that keeps a kind
  of region injected mentions become (signature, handwriting, photo, face, logo, barcode, stamp, illegible) is
  refused (exit 1, `E_CONFIG_INVALID`).  NER is switched off (it would load for nothing).
- Verification counts every residual hit against the release, but re-plans fewer of them: all of them in
  `oracle_text` (as in `e2e`), only removed values that are still readable in `oracle_mentions` and
  `oracle_entities` (anything else it finds was not injected), none in `oracle_regions` (the plan is applied
  as given).
- Ids: `text.json` lists tool 2's own token ids in every mode (the manifest's schema requires them).  From
  `oracle_mentions` on, `mentions.json`, `entities.json`, `plan.json` and `pseudonyms.json` name the injected
  mention ids (a mention inside another keeps its own entry); from `oracle_entities` on the clusters are the
  injected ones, under their ids; in `oracle_regions` `plan.json` repeats the injected plan regions (ids and
  polygons) with what was done to each (manifest.json gives each injected region the profile's action on its
  kind, whether or not the plan applies it).  The notes in run_info.json count what was injected and what could
  not be used, never text.

Exit codes: `0` output.pdf written (released or not), `1` no output: nothing was built, the injected inputs
could not be read (`E_CONFIG_INVALID`), or the files could not be written (run_info.json still says why),
`2` usage error (also an oracle mode without a folder of injected inputs, or `e2e` with one), `3` unsupported
mode.  output.pdf is written last, so it exists only together with every other file.

| Environment | Effect |
|---|---|
| `MLREDACT_CONFIG` | deployment YAMLs (path-separator separated), applied before `--config` |
| `MLREDACT_SURROGATE_KEY` | the surrogate key (or `surrogate.key_file`); without one the public development key is used and run_info.json says so |
| `MLREDACT_MODELS_DIR` | models folder; when unset and the configured `.models` is not in the current folder (the benchmark starts the adapter from its run folder), the backend's own `.models` is used |
| `RB_MLREDACT_PYTHON` | benchmark side: this backend's Python, used by the candidate manifest |

On a machine without Tesseract, point `MLREDACT_CONFIG` at `configs/runtime/laptop-no-engine-b.yaml`
(absolute path); results then differ from runs with engine B.

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
  config/      pydantic schema, layered loader, built-in profiles (client, broad, claimant_focused, maximal)
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
  api/         HTTP API for the frontend: jobs, queue, signed links (FastAPI)
  bench/       benchmark adapter: stage files from a job trace, injected stage inputs (evaluator only)
  cli/         Typer CLI (run, bench, serve, selfcheck, ...)
tools/synthdocs/  synthetic test documents
docs/adr/         architecture decision records
```

## Determinism

Outputs are byte-identical for the same input, configuration, models, software and **runtime
profile** (e.g. `linux-x86_64-v3`).  Pin `runtime.expected_profile` in production; workers refuse to
start on a mismatch.  Thread counts are fixed per process, batch sizes are fixed (1), IDs are
derived from positions, and every collection is emitted in canonical order.
