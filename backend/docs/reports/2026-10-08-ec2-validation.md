# EC2 validation run: full test suite with all models (2026-10-08)

- Scope: first run of the complete pipeline, with every default model and Tesseract, on Linux
hardware.  Development only, synthetic documents only.
- Runs: run 1 on 2026-10-05, run 2 (after one fix) on 2026-10-08.
- Raw logs: [2026-10-08-ec2-logs/](2026-10-08-ec2-logs/)

## 1. Summary


|                                  | Run 1 (2026-10-05)                                | Run 2 (2026-10-08) |
| -------------------------------- | ------------------------------------------------- | ------------------ |
| `mlredact selfcheck`             | pass (after installing two system packages, §4.2) | pass               |
| Fast suite (`make test`)         | 230 passed                                        | 230 passed         |
| Determinism (`make determinism`) | not run                                           | 1 + 8 passed       |
| Slow suite (`make test-slow`)    | 21 passed, **1 failed**                           | **22 passed**      |


- **All tests pass** on the instance after one code fix.
- **Fixed bug:** OCR engine B (Tesseract) returned no words at all, so the second OCR reader was
silently doing nothing (§4.1).
- **Determinism holds on this machine:** repeated runs, different worker counts and windowed vs.
whole-document NER all give identical results.  Reference outputs for comparing *across* machines
are not recorded yet (§6).
- **Memory:** the largest single process peaked at **10.75 GiB**.  A 16 GiB instance is enough but
leaves about 4 GiB of headroom (§5).
- **Speed:** the slow suite takes **~38 min** on 4 vCPU, roughly double the 10–20 min in MODELS.md (§5).
- **Docs:** MODELS.md is missing two system packages and one model from its setup steps (§4.2).



## 2. Environment


|                 |                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------- |
| Instance        | `m7i.xlarge`, on-demand, `ap-south-1` (Mumbai), AZ `ap-south-1b`                                          |
| CPU             | Intel Xeon Platinum 8488C (Sapphire Rapids), 4 vCPU = 2 cores × 2 threads                                 |
| Instruction set | AVX2, AVX-512F, AVX-512 VNNI, AMX → runtime profile `linux-x86_64-v4`                                     |
| Memory / disk   | 15 GiB RAM / 50 GiB gp3 encrypted (5.9 GiB used)                                                          |
| OS              | **Ubuntu 26.04 LTS**, kernel 7.0.0-1014-aws (MODELS.md says 24.04; the launch wizard's default was newer) |
| Python / uv     | CPython 3.12.15 (installed by uv) / uv 0.12.23                                                            |
| Tesseract       | 5.5.0 (`tesseract-ocr 5.5.0-1build1`), leptonica 1.86.0                                                   |
| Key libraries   | onnxruntime 1.30.0, opencv 5.0.0, numpy 2.5.3, pdfium 153.0.7999.0, rapidocr 3.9.2, tokenizers 0.22.2     |
| Config hash     | `96d1ff0146d456f3c8caccf53ec393238576689abf7f357aca209d0f6b0d5657`                                        |


Full versions: [run2-selfcheck.json](2026-10-08-ec2-logs/run2-selfcheck.json).

### Models (profile `broad`, all SHA-256 verified by `mlredact models fetch` and `selfcheck`)


| Model                         | Size on disk | Role                               |
| ----------------------------- | ------------ | ---------------------------------- |
| `privacy_filter_q4`           | 902 MB       | NER N1                             |
| `gliner_pii_base`             | 643 MB       | NER N2                             |
| `ppocrv6_rec_medium`          | 74 MB        | OCR recognition (primary)          |
| `ppocrv6_rec_small`           | 21 MB        | OCR re-read (verification V4)      |
| `tessdata_best_eng`           | 15 MB        | Tesseract English model (engine B) |
| `ppocrv6_det_small`           | 9.5 MB       | OCR text detection                 |
| `pplcnet_textline_ori_server` | 6.5 MB       | text-line orientation              |
| `yunet_face_2023mar`          | 232 KB       | face detection                     |


Optional models not fetched: `ppocrv6_det_medium`, `gliner_pii_large`.

## 3. What was run


| Command                                      | What it covers                                                                                                                      | Run 2 result | Wall time   | Peak RSS* |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------ | ----------- | --------- |
| `mlredact selfcheck`                         | model hashes, 17 resources, runtime profile                                                                                         | pass         | –           | –         |
| `pytest tests/integration/test_tesseract.py` | engine B on the real binary (re-test of the run 1 failure)                                                                          | 1 passed     | 4.6 s       | –         |
| `make test`                                  | unit, property, security, determinism tests without models                                                                          | 230 passed   | 5.4 s       | 0.51 GiB  |
| `make determinism`                           | hash-seed test + 8 slow tests: byte-identical reruns, worker-count independence, NER windowing, deterministic surrogates, Tesseract | 1 + 8 passed | 20 min 40 s | 7.38 GiB  |
| `make test-slow`                             | all 22 end-to-end and model-backed tests (below)                                                                                    | 22 passed    | 37 min 40 s | 10.75 GiB |


 Peak resident memory of the largest single process (GNU `time -f %M`), not the sum across
processes.

The 22 slow tests:

- **Models:** YuNet loads and finds no faces on a text page.  GLiNER finds people, places and
organisations.  The Privacy Filter finds identifiers, respects the allowlist, and gives identical
results windowed and whole-document.  Tesseract reads a degraded scanned letter.
- **Pipeline, release and recall:** a scanned letter is released and its planted PII is
unrecoverable by an independent OCR reader.  A digital letter's text layer is rebuilt without PII.
Signatures, barcodes, stamps, handwriting and show-through are removed.  TIFF scans are accepted.
- **Pipeline, determinism:** output is byte-identical across runs and independent of worker count.
Surrogate mode is deterministic and consistent across documents in a scope.
- **Pipeline, profiles and modes:** surrogate, label and blackout modes.  The maximal profile shifts
dates while preserving intervals.  The claimant-focused profile keeps the treating professional.
- **Pipeline, safety:** encrypted and non-PDF inputs are quarantined.  The manifest contains no
document text.  The sealed sensitive manifest opens only for the evaluator.  The output opens in an
independent PDF engine.



## 4. Findings



### 4.1 Bug (fixed): Tesseract produced no words

Run 1 failure: `test_tesseract_reads_the_scanned_letter_deterministically`, where engine B returned an
empty word list.

- **Cause:** [src/mlredact/ocr/tesseract.py](../../src/mlredact/ocr/tesseract.py) passed `tsv` as
the last argument.  Tesseract treats that as the name of a config file and looks for it in
`<tessdata-dir>/configs/tsv`.  The hash-pinned model directory holds only `eng.traineddata`, so
Tesseract printed `read_params_file: Can't open tsv`, exited 0 and wrote plain text.  The TSV
parser found no word rows.
- **Effect:** wherever engine B was enabled it was silently a no-op, in the primary pass and in the
V4 re-read.  No error was raised, so nothing reported it.  The ADR 0005 benefits from engine B
(correcting uncertain engine-A tokens, adding missed words) were not being delivered.
- **Fix:** request TSV output explicitly with `-c tessedit_create_tsv=1`.  This does not depend on
any config file and leaves the pinned model directory untouched.  Verified by hand (141 words
from the test page) and by the test, which now passes.
- **Not checked:** whether the Debian Tesseract in the Docker image found its system `configs/tsv`
and so worked before the fix.  The fix is correct either way.
- **Follow-up:** add a guard so that an engine returning zero words on a page with ink is reported,
not silently accepted.  A silent no-op reader is the failure this bug shows is possible.



### 4.2 Setup gaps in [MODELS.md](http://MODELS.md)

A fresh Ubuntu instance following MODELS.md §3 fails `selfcheck` with
`ImportError: libGL.so.1` (from OpenCV).  Tesseract is also missing.  Before running
`uv sync`, the instance also needs:

```bash
sudo apt-get install -y tesseract-ocr libgl1
```

`tesseract-ocr` is in [docker/Dockerfile.cpu](../../docker/Dockerfile.cpu) but not in MODELS.md.
`libgl1` is in neither.  Also:

- MODELS.md §1 does not list `tessdata_best_eng`, which the `broad` profile fetches.
- MODELS.md says Ubuntu 24.04.  The instance runs 26.04, with Tesseract 5.5.0, while the Docker
image (Debian bookworm) ships an older Tesseract.  Engine B output may therefore differ between
this instance and the Docker image.  Pin one for anything that compares outputs.



### 4.3 Determinism

Every determinism check passed.  These tests compare runs **on the same machine**: two runs of the
same input, 1 worker vs. several, and windowed vs. whole-document NER.  They show the pipeline is
deterministic on `linux-x86_64-v4`.  They do not store reference hashes, so they cannot yet show that
a different machine of the same profile, or the production image, produces the same bytes (§6).

## 5. Resources and timing

- **Memory:** peak 10.75 GiB in the slow suite and 7.38 GiB in the determinism tests (largest
process).  The NER models alone are ~2.5 GB (MODELS.md).  The rest is OCR, rendering and the test
harness holding several documents and redactors.  Do not run two model-backed jobs at once on a
16 GiB instance.  For `gliner_pii_large` (+~2.5 GB) use 32 GiB (`m7i.2xlarge`).
- **Time:** slow suite 37 min 40 s in run 2 and 43 min 28 s in run 1.  Run 1 was slower even though
its Tesseract was doing nothing.  The likely cause is first-use effects on a new disk (EBS blocks
load lazily on first read); this was not investigated.  MODELS.md's 10–20 min estimate is low for
4 vCPU, which here is only 2 physical cores.  An 8 vCPU instance should cut this substantially but
was not measured.
- **Cost:** `m7i.xlarge` on-demand in Mumbai is roughly ₹19/hour plus ~₹400/month for the 50 GB disk
(estimates, excluding GST).  One full validation (setup + all suites) is about 1.5 hours of
instance time.



## 6. Open items

1. **Reference outputs across machines:** record output hashes for a fixed set of synthetic
  documents on this instance type, and compare them in CI or on the production image.  This is the
   remaining part of the determinism goal.
2. **Zero-word guard for second readers** (§4.1 follow-up).
3. **Update MODELS.md** with the packages, the missing model, realistic timings and the OS note
  (§4.2, §5).
4. **Update ADR 0005's status:** engine B is now validated on the Linux instance.
5. **Model benchmarks:** none were run.  This report covers correctness tests only.  Recall,
  precision and per-page speed for the Privacy Filter, GLiNER and any candidate models still need
   the evaluator on a synthetic corpus.



## 7. Access and security notes

- **Access:** IAM user with an EC2-only policy limited to `ap-south-1`.  Console sign-in only, no
access keys.
- **SSH problems during the run:** the developer's public IP changed four times across two ISPs.
The policy lacks `ec2:ModifySecurityGroupRules` and `ec2:RevokeSecurityGroupEgress`, so existing
rules could not be edited.  New rules could only be added.
- **Current state:** security group `sg-0ea7c001002d35f11` has an SSH rule from `0.0.0.0/0`,
added to finish this run, plus stale per-IP rules.  Password login is disabled
(`PasswordAuthentication no`), so SSH needs the private key.  **Remove the** `0.0.0.0/0` **rule and the
stale rules** once the policy allows it.  The client has been asked for `ec2:ModifySecurityGroupRules`.
Session Manager would remove the need for any inbound rule.
- Only synthetic documents were used on the instance.

