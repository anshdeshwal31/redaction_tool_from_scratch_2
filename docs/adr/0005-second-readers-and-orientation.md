# ADR 0005 — Second readers (embedded text, Tesseract), form geometry and page orientation

- Status: accepted (engine B to be validated on the Linux instance; it is not installed on the dev laptop)
- Date: 2026-10-04

## Context

Plan §6 calls for more than one reading of every page: engine A (PP-OCRv6), engine B (Tesseract 5)
and engine D (the PDF's own text layer), fused so that tokens found by only one engine are kept.  It
also calls for page normalisation (orientation, deskew) and, in §7.2 R3, for label→value extraction
across table cells and lines.

## Measurements

* **Orientation and skew** (synthetic letter, 300 dpi, engine A alone): text similarity to the
  ground truth was 99–100% at 0°, 90°, 180° and 270°, and at 5° and 10° skew.  The claimant's name
  was found every time.  Per-line quad warping plus the text-line orientation classifier already
  handle both.
* **Embedded text**: PDFium's character boxes map exactly onto the canonical raster through
  `FPDF_PageToDevice` (60 pt → 250 px at 300 dpi).

## Decision

* **No page-level orientation/deskew stage.**  Engine A reads rotated and skewed pages.  Mentions on
  lines more than 25° from horizontal are rendered as blackout, because surrogates and labels are
  only typeset horizontally.  Revisit if the evaluator attributes misses to orientation on real
  scans.
* **Engine D** (`ingest/embedded.py`, `ocr/fusion.py`): words of the PDF text layer, kept only when
  there is ink under them (paper level measured around the word) and the text is plausibly encoded
  (no private-use or replacement characters).
* **Engine B** (`ocr/tesseract.py`): the system Tesseract 5 binary as a subprocess.
  * One OpenMP thread, fixed DPI, and the hash-pinned `tessdata_best` English model.
  * The binary's version is recorded in the runtime profile.
  * It runs in both the primary pass and the V4 re-read of the output.
  * The service refuses to start if engine B is enabled and the binary is missing.
* **Fusion rule (B and D)**:
  * A second reading of an engine-A token replaces A's text only when A was uncertain (weakest
    character < 0.9) and the second reader is confident (≥ 0.9).
  * Otherwise the second reading is an alternate, which propagation searches.
  * Confident words A did not read become new tokens: all embedded words; Tesseract words ≥ 70.
  * Tokens explain ink (ADR 0004) only if plausible, so a confident garbage read cannot hide a mark.
* **Form geometry** (`detect/rules/labels.py`): a label with nothing after it on its line takes its
  value from a separate cell on the same row, else from the line directly below in its column.
  * Only for short label lines or labels ending in ":".
  * The value must still parse as the label's kind.

## Consequences

* Born-digital pages get exact text where OCR is unsure, and print too small for OCR still reaches
  detection (tested with 3-pt text).
* A broken font encoding cannot overwrite a confident OCR reading.
* Tesseract adds roughly 5–10 s per page on CPU.  It is disabled on the dev laptop by the local
  override, and the end-to-end Tesseract test skips where the binary is absent.
