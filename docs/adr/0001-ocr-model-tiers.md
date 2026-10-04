# ADR 0001 — OCR engine A model tiers and batch size

- Status: accepted (revisit when the external evaluation reports OCR-attributable misses)
- Date: 2026-10-04

## Context

Engine A is PP-OCRv6 (Apache-2.0) via ONNX Runtime.  PP-OCRv6 ships tiny/small/medium tiers.
Measured on the development profile (`windows-x86_64-v3`, Ryzen 5 5600H, 1 thread per process,
300 dpi A4 synthetic IME letter, clean and degraded scans):

| detector / recogniser | seconds / page | text similarity |
|---|---|---|
| medium (full res) / medium | ~60 | 0.989 |
| medium / small | ~40 | 0.987 |
| small / medium | ~22 | 0.989 |
| small / small | ~12 | 0.987 |

Detection at full resolution dominated runtime for the medium detector (43 s single-threaded);
reducing the detector input to a 2000 px long side cut that to 12 s but would cost small-text
recall.  All tiers read the synthetic pages almost perfectly, so the synthetic set cannot separate
them on accuracy; PaddleOCR's own benchmarks place medium recognition several points above small.

Recognition batching (RapidOCR default: 6 lines, padded to the widest) makes a line's result depend
on the other lines on the page.

## Decision

* Primary reading: **small detector at full canonical resolution + medium recogniser**.
  Recognition quality dominates recall; full-resolution detection protects small print (footers,
  patient labels).  Both are configurable (`ocr.engine_a.*`).
* Verification re-read (V4): small/small — it is a check over the output, not the primary reading,
  and an independent model configuration adds diversity.
* Batch size 1 everywhere (detector, orientation classifier, recogniser): results are invariant to
  the composition of the page, at a small CPU cost.
* Low-score lines are never dropped (the library's `text_score` filter is bypassed); confidence is
  carried per character into the uncertain-neighbour rule instead.

## Consequences

* CPU throughput comes from page-level process parallelism, not intra-op threads.
* Revisit with real scans: if the evaluator attributes misses to detection, move the detector to
  medium (GPU profile) or add Tesseract as engine B (planned, Phase 2).
