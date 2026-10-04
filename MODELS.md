# Models: what to host, where, and how

mlredact uses only pinned, locally hosted models.  Nothing is called over the internet at run time.
Every model file is identified by name + SHA-256 in
[src/mlredact/registry/models.yaml](src/mlredact/registry/models.yaml) and is verified before every
use.  A missing or altered file stops the job (`E_MODEL_MISSING` / `E_MODEL_HASH_MISMATCH`); the
pipeline never silently runs without a model it is configured to use.

## 1. The models

| id | used for | file size | RAM in use | licence | needed by default |
|---|---|---|---|---|---|
| `ppocrv6_det_small` | OCR: text detection | 9.5 MB | small | Apache-2.0 | yes |
| `ppocrv6_rec_medium` | OCR: text recognition (primary) | 74 MB | small | Apache-2.0 | yes |
| `ppocrv6_rec_small` | OCR: re-reading the output (verification V4) | 21 MB | small | Apache-2.0 | yes |
| `pplcnet_textline_ori_server` | OCR: text-line orientation | 6.5 MB | small | Apache-2.0 | yes |
| `gliner_pii_base` | NER (N2): GLiNER-PII base, fp32 ONNX | 665 MB | ~1 GB | Apache-2.0 | yes |
| `privacy_filter_q4` | NER (N1): OpenAI Privacy Filter, int4 weight-only ONNX | 917 MB | ~1.2–1.5 GB | Apache-2.0 | yes |
| `yunet_face_2023mar` | face detection (photos of people, ID-card photos) | 233 KB | small | MIT | yes |
| `ppocrv6_det_medium` | OCR detector alternative (GPU profile / experiments) | 60 MB | small | Apache-2.0 | no |
| `gliner_pii_large` | NER alternative: GLiNER-PII large | 1.76 GB | ~2.5 GB | Apache-2.0 | no |
| `tessdata_best_eng` | OCR engine B: Tesseract 5 LSTM English model | 15 MB | small | Apache-2.0 | yes |

OCR engine B also needs the **Tesseract 5 binary** (`apt-get install tesseract-ocr`).  Its version is
recorded in each manifest's runtime profile; the service refuses to start if engine B is enabled and
the binary is missing.

* The OCR models and YuNet are small and run in every page-worker process.  They are fine on a laptop.
  Barcodes are read by the `zxing-cpp` library (no model file).
* The two **NER models** run once per service process (not per worker, not per document): ~2.5 GB
  RAM in total, ~35 s to load and ~20 s for the first inference after loading.  They are the reason
  for the hosting arrangement below.
* Speed on CPU (4 threads): the Privacy Filter takes ~1.3 s per call plus ~12 s per 1,000 tokens on
  long documents; GLiNER ~0.8 s per 300-word window.  See
  [docs/adr/0002-ner-ensemble.md](docs/adr/0002-ner-ensemble.md).

Download and verify exactly the models a profile needs:

```bash
uv run mlredact models fetch --profile broad      # into .models/ (or $MLREDACT_MODELS_DIR)
uv run mlredact selfcheck                         # verifies models, resources, runtime profile
```

`mlredact models fetch` with no arguments downloads *every* registered model, including the optional
1.76 GB `gliner_pii_large`.

## 2. Where to host them

### Development: one EC2 instance, used through VS Code Remote-SSH

The whole application (code, tests, all models) runs on the instance.  The laptop is only the
editor.  This needs no code changes and gives the same results production will give, provided the
CPU type matches (below).

| setting | recommendation | why |
|---|---|---|
| Region | `ap-southeast-2` (Sydney) | Australian health and legal data should stay in Australia (Privacy Act APP 8), and production will need this anyway |
| Instance | x86-64, 8–16 vCPU, 32–64 GiB RAM; e.g. `m7i.2xlarge` (8 vCPU, 32 GiB) or `m7i.4xlarge` (16 vCPU, 64 GiB) | NER ~2.5 GB + OCR workers ~0.4 GB each; 64 GiB matches the plan's CPU node (plan §16) |
| CPU family | **the same family you will use in production** | outputs are byte-identical only on the same instruction-set level (AVX2 vs AVX-512 change floating-point results) |
| OS | Ubuntu 24.04 LTS | the dev container and Dockerfile are Linux; Windows is not a determinism target |
| Disk | 50 GB gp3, encrypted | models ~1.8 GB, Python environment, Docker images, test outputs |
| GPU | none for now | only needed for the generative tiers (plan phase P5) |

Security baseline for the instance:

* SSH key authentication only; security-group inbound rule: port 22 **from your IP only**.  No other
  inbound ports; never expose a model or application port publicly.
* IMDSv2 required; encrypted EBS; no IAM role unless needed.
* **Synthetic documents only** on the development instance.  Real documents need the production
  controls (plan §14).
* Stop the instance when not in use; you pay only for running hours (plus the disk).

### Production

Run the application on the same instance family as development, inside a private subnet with **no
internet egress**:

* Bake the models into the image at build time (the Dockerfile's `models` stage) or mount them
  read-only from a volume populated by `mlredact models fetch --profile <profile>`; they are
  hash-verified at every start either way.
* Pin the runtime profile: set `runtime.expected_profile` in the deployment config to the value
  `mlredact selfcheck` reports on that instance type (e.g. `linux-x86_64-v4` on AVX-512 CPUs).
  Workers refuse to start on a different CPU level.
* Planned (phase P6, not built yet): the NER models as a separate model service inside the same
  private network, so one copy serves many worker hosts.  Until then each service process loads
  its own copy.

## 3. Setting up the development instance

1. Launch the instance (settings above) with your key pair; note its public DNS name.
2. On the laptop, add an SSH host alias to `~/.ssh/config`:

   ```
   Host mlredact-dev
       HostName <instance public DNS>
       User ubuntu
       IdentityFile ~/.ssh/<your-key>.pem
   ```

3. Copy the project to the instance.  Nothing is committed yet, so either commit and push to a
   private repository and clone it there, or copy the working tree:

   ```bash
   rsync -av --exclude .venv --exclude .models --exclude __pycache__ ./ mlredact-dev:~/mlredact/
   ```

4. On the instance:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh     # uv (installs the pinned Python 3.12 itself)
   sudo apt-get update && sudo apt-get install -y make tesseract-ocr
   cd ~/mlredact
   uv sync --locked
   uv run mlredact models fetch --profile broad        # ~1.8 GB, SHA-256 verified
   uv run mlredact selfcheck                           # note the runtime_profile it prints
   make test                                           # fast suite
   make test-slow                                      # end-to-end with NER (~10-20 min)
   ```

5. In VS Code: *Remote-SSH: Connect to Host… → mlredact-dev*, open `~/mlredact`.

## 4. The laptop arrangement (NER and Tesseract switched off)

On the laptop the NER models and OCR engine B (Tesseract, not installed there) are **off**:

* The NER weights are parked outside the project in `~/.mlredact/parked-models/` (they are no longer
  in `.models/`).  A run with the normal configuration therefore stops with `E_MODEL_MISSING`; it
  never runs with less detection than configured.
* Pipeline runs on the laptop use a local override kept **outside the repository**,
  `~/.mlredact/laptop-no-ner.yaml`:

  ```yaml
  detection:
    ner:
      gliner:
        enabled: false
      privacy_filter:
        enabled: false
  ocr:
    engine_b:
      enabled: false
  ```

  ```bash
  uv run mlredact selfcheck --config ~/.mlredact/laptop-no-ner.yaml
  uv run mlredact run input.pdf --out out/ --dev-key --config ~/.mlredact/laptop-no-ner.yaml
  ```

  Detection is then rules + propagation only: **lower recall**.  Laptop output is for development and
  must never be used for real documents.
* `make test` (fast suite) never loads NER and runs normally on the laptop.  The end-to-end suite can
  run on the laptop with the same override (NER and Tesseract tests are skipped):

  ```bash
  MLREDACT_TEST_CONFIG=~/.mlredact/laptop-no-ner.yaml make test-slow
  ```

  The full `make test-slow` (with NER and Tesseract) belongs on the instance.
* To restore NER on the laptop, move the two folders back (or re-download them, hash-verified):

  ```bash
  mv ~/.mlredact/parked-models/* .models/          # or: uv run mlredact models fetch --profile broad
  ```
