"""Command-line interface.

Output files have fixed names (``redacted.pdf``, ``manifest.json``, ``run.json``,
``quarantine.json``, and ``held-back.NOT-RELEASED.pdf`` with ``--keep-held-back``): input
filenames often contain names and are never echoed or reused.
Exit codes: 0 released, 2 quarantined, 1 usage/environment error.  ``bench`` (the benchmark
adapter, for the evaluator only) has its own files and exit codes, described in its help.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from mlredact import __version__
from mlredact.config.loader import builtin_profiles, config_hash, load_config
from mlredact.core.canonical import canonical_json_pretty
from mlredact.core.errors import EnvironmentErrorMl, MlredactError, ReasonCode
from mlredact.core.types import EntityType, JobStatus
from mlredact.security.exceptions import install_excepthook, sanitize
from mlredact.security.logging import configure_logging
from mlredact.security.sensitive import Sensitive

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Local, deterministic PII redaction.")
models_app = typer.Typer(no_args_is_help=True, help="Model artefacts (development / image build).")
app.add_typer(models_app, name="models")
keys_app = typer.Typer(no_args_is_help=True, help="Key material for sealed sensitive manifests.")
app.add_typer(keys_app, name="keys")
sensitive_app = typer.Typer(no_args_is_help=True, help="Sealed sensitive manifests (evaluator side).")
app.add_typer(sensitive_app, name="sensitive")

# The redacted copy of a quarantined document: it failed verification, so its name must never
# pass for a release.
HELD_BACK_PDF = "held-back.NOT-RELEASED.pdf"
# Benchmark modes ``bench`` runs.  The oracle modes (injected stage inputs) come in the next phase.
BENCH_MODES = ("e2e",)


def _fail(exc: BaseException) -> None:
    info = sanitize(exc)
    typer.echo(f"error: {info.code.value} ({info.error_type} at {info.where})", err=True)
    raise typer.Exit(1)


def _quarantine_record(manifest: Mapping[str, Any]) -> bytes:
    return canonical_json_pretty(
        {
            "status": "quarantined",
            "reasons": manifest["reasons"],
            "input_sha256": manifest["job"]["input_sha256"],
        }
    )


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    if os.name != "nt":
        tmp.chmod(0o600)
    tmp.replace(path)


def _load_seeds(path: Path | None) -> list[Any]:
    from mlredact.detect.engine import Seed

    if path is None:
        return []
    raw = json.loads(path.read_text("utf-8"))
    seeds = []
    for item in raw:
        seeds.append(Seed(EntityType(item["type"]), Sensitive(str(item["value"]))))
    return seeds


@app.command()
def run(
    input_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Input PDF, TIFF, PNG or JPEG")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")] = Path("out"),
    profile: Annotated[str, typer.Option(help="Policy profile")] = "client",
    config: Annotated[list[Path] | None, typer.Option("--config", "-c", help="Deployment YAML (repeatable)")] = None,
    style: Annotated[str | None, typer.Option(help="Override render style: surrogate | label | blackout")] = None,
    workers: Annotated[int | None, typer.Option(help="Worker processes")] = None,
    seeds: Annotated[Path | None, typer.Option(help="JSON list of known identifiers [{type, value}]")] = None,
    scope_id: Annotated[
        str | None, typer.Option(help="Surrogate scope (e.g. matter id): consistent surrogates across documents")
    ] = None,
    dev_key: Annotated[
        bool, typer.Option("--dev-key", help="Use the public development surrogate key (never in production)")
    ] = False,
    sensitive_to: Annotated[
        Path | None,
        typer.Option(help="Evaluator public key file: also write the sealed sensitive manifest (original text)"),
    ] = None,
    keep_held_back: Annotated[
        bool,
        typer.Option(
            "--keep-held-back",
            help=f"If the document is quarantined, also write its redacted copy as {HELD_BACK_PDF} "
            "(testing and review only: it failed verification and is not a release)",
        ),
    ] = False,
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Redact INPUT and write the release artefacts (or a quarantine record) to --out."""
    install_excepthook()
    configure_logging(log_level)
    try:
        from mlredact.pipeline.runner import Redactor

        overrides: dict[str, Any] = {}
        if style:
            overrides["redaction"] = {"style": style}
        if workers:
            overrides["runtime"] = {"workers": workers}
        if dev_key:
            overrides["surrogate"] = {"allow_development_key": True}
        cfg = load_config(profile, config or [], overrides)
        data = input_path.read_bytes()
        seed_list = _load_seeds(seeds)
        recipient = None
        if sensitive_to is not None:
            from mlredact.manifest.sensitive import load_public_key

            recipient = load_public_key(sensitive_to.read_text("utf-8"))
        with Redactor(cfg, log_level=log_level) as redactor:
            result = redactor.run(
                data, seeds=seed_list, scope_id=scope_id, sensitive_to=recipient, keep_held_back=keep_held_back
            )
    except MlredactError as exc:
        _fail(exc)
        return
    _write(out / "manifest.json", canonical_json_pretty(result.manifest))
    _write(out / "run.json", canonical_json_pretty(result.run_record))
    if result.sensitive is not None:
        _write(out / "sensitive-manifest.sealed.json", result.sensitive)
    if result.status is JobStatus.RELEASED and result.pdf is not None:
        _write(out / "redacted.pdf", result.pdf)
        typer.echo(f"released: {out / 'redacted.pdf'}")
        raise typer.Exit(0)
    _write(out / "quarantine.json", _quarantine_record(result.manifest))
    typer.echo(f"quarantined: {', '.join(result.manifest['reasons']) or 'verification failed'}", err=True)
    if result.held_back_pdf is not None:
        _write(out / HELD_BACK_PDF, result.held_back_pdf)
        typer.echo(f"held-back copy (not released): {out / HELD_BACK_PDF}", err=True)
    raise typer.Exit(2)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.command("bench-version")
def bench_version() -> None:
    """Print the version the benchmark records for bench: mlredact's version and commit."""
    from mlredact.bench.environment import candidate_version

    typer.echo(candidate_version())


# One string per paragraph: the help renderer keeps line breaks inside a paragraph.
_BENCH_HELP = (
    "Benchmark adapter, for the evaluator only (not production): redact --input into --workdir.\n\n"
    "Writes output.pdf and the benchmark's stage files (text.json, mentions.json, entities.json, plan.json, "
    "pseudonyms.json, run_info.json; they hold document text) next to manifest.json, run.json and, when the "
    "document is not released, quarantine.json.  Output the tool does not release (held back by its "
    "verification, or built before a later step failed) is still written as output.pdf, and run_info.json "
    "records the verdict.\n\n"
    "Exit codes: 0 output.pdf written (released or not), 1 no output (nothing was built, or the files could not "
    "be written), 2 usage error, 3 unsupported mode."
)


@app.command(help=_BENCH_HELP)
def bench(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, dir_okay=False, readable=True, help="The benchmark's input.pdf")
    ],
    workdir: Annotated[Path, typer.Option("--workdir", file_okay=False, help="Empty folder for the outputs")],
    mode: Annotated[str, typer.Option(help="Benchmark mode (only e2e so far)")] = "e2e",
    inject_dir: Annotated[
        str, typer.Option("--inject-dir", help="Injected stage inputs (oracle modes); empty for none")
    ] = "",
    seed: Annotated[int | None, typer.Option(help="Benchmark seed (recorded only: runs are deterministic)")] = None,
    scope_id: Annotated[
        str, typer.Option("--scope-id", help="Surrogate scope, e.g. the matter id (empty: the document itself)")
    ] = "",
    profile: Annotated[str, typer.Option(help="Policy profile")] = "client",
    config: Annotated[
        list[Path] | None,
        typer.Option(
            "--config", "-c", exists=True, dir_okay=False, help="Deployment YAML (repeatable; after MLREDACT_CONFIG)"
        ),
    ] = None,
    workers: Annotated[int | None, typer.Option(help="Worker processes")] = None,
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Benchmark adapter for the evaluator (see ``_BENCH_HELP``); production uses ``run``."""
    if mode not in BENCH_MODES:
        typer.echo(f"error: mode {mode} is not supported yet: the oracle modes come in the next phase", err=True)
        raise typer.Exit(3)
    if inject_dir:
        raise typer.BadParameter("mode e2e takes no injected inputs", param_hint="--inject-dir")
    from mlredact.bench.environment import CONFIG_ENV, env_config_files

    env_files = env_config_files()
    if not all(p.is_file() for p in env_files):
        raise typer.BadParameter("a file it names does not exist", param_hint=CONFIG_ENV)
    install_excepthook()
    configure_logging(log_level)
    from mlredact.bench.artifacts import run_info, stage_files, unreleased_note
    from mlredact.bench.environment import candidate_version, models_dir_fallback, surrogate_key_source

    started = _utc_now()
    version = candidate_version()
    files = [*env_files, *(config or [])]
    overrides: dict[str, Any] = {"surrogate": {"allow_development_key": True}}
    if workers:
        overrides["runtime"] = {"workers": workers}
    settings: dict[str, Any] = {"profile": profile}
    models: list[dict[str, Any]] = []
    notes: list[str] = []

    def finish(released: bool, reasons: list[str], *outcome: str) -> None:
        info = run_info(
            version=version,
            config=settings,
            seed=seed,
            models=models,
            started_at=started,
            finished_at=_utc_now(),
            notes=[*notes, *outcome],
            released=released,
            checks_failed=reasons,
        )
        _write(workdir / "run_info.json", canonical_json_pretty(info))

    try:
        from mlredact.pipeline.runner import Redactor

        cfg = load_config(profile, files, overrides)
        fallback = models_dir_fallback(cfg.runtime.models_dir)
        if fallback is not None:  # the benchmark starts us from its run folder
            overrides["runtime"] = {**overrides.get("runtime", {}), "models_dir": str(fallback)}
            cfg = load_config(profile, files, overrides)
            notes.append("models from the backend's own .models folder")
        settings = {
            "profile": profile,
            "render_style": cfg.redaction.style,
            "config_hash": config_hash(cfg),
            "scope": "caller" if scope_id else "document",
            "workers": cfg.runtime.workers,
            "engine_b_enabled": cfg.ocr.engine_b.enabled,
        }
        key = surrogate_key_source(cfg)
        if key == "development":
            notes.append("public development surrogate key: no MLREDACT_SURROGATE_KEY or key file is configured")
        elif key is not None:
            notes.append(f"configured surrogate key ({key})")
        data = input_path.read_bytes()
        with Redactor(cfg, log_level=log_level) as redactor:
            result = redactor.run(data, scope_id=scope_id or None, trace=True)
    except Exception as exc:  # whatever stopped the job, the evaluator still gets run_info.json
        finish(False, [sanitize(exc).code.value])
        _fail(exc)
        return
    released = result.status is JobStatus.RELEASED
    reasons = [] if released else list(result.manifest["reasons"])
    models = list(result.manifest["provenance"]["models"])
    trace = result.trace
    output = result.pdf if result.pdf is not None else (trace.pdf if trace is not None else None)
    handed_over = False
    try:
        _write(workdir / "manifest.json", canonical_json_pretty(result.manifest))
        _write(workdir / "run.json", canonical_json_pretty(result.run_record))
        if not released:
            _write(workdir / "quarantine.json", _quarantine_record(result.manifest))
        if output is not None and trace is not None:
            # The stage files are serialised before the first is written, and output.pdf is written last: it
            # exists only when every other file does, so the benchmark never grades an output without its
            # stage files and the tool's verdict.
            stages = stage_files(trace, text_layer=cfg.ocr.embedded_text)
            blobs = {name: canonical_json_pretty(document) for name, document in stages.items()}
            for name, blob in blobs.items():
                _write(workdir / name, blob)
            finish(released, reasons, *([] if released else [unreleased_note(reasons)]))
            _write(workdir / "output.pdf", output)
            handed_over = True
        else:
            finish(False, reasons, "the pipeline built no output")
    except Exception as exc:  # a file could not be built or written: no output.pdf, and run_info.json says why
        finish(False, [*reasons, sanitize(exc).code.value], "the adapter could not write its files: no output.pdf")
        _fail(exc)
        return
    if not handed_over:
        typer.echo(f"no output: {', '.join(reasons) or 'no reason recorded'}", err=True)
        raise typer.Exit(1)
    typer.echo("released: output.pdf" if released else f"not released ({', '.join(reasons)}): output.pdf written")


@app.command()
def selfcheck(
    profile: Annotated[str, typer.Option()] = "client",
    config: Annotated[list[Path] | None, typer.Option("--config", "-c", help="Deployment YAML (repeatable)")] = None,
) -> None:
    """Validate configuration, runtime profile, resources, model artefacts and engine binaries."""
    install_excepthook()
    try:
        from mlredact.ocr.tesseract import binary_version
        from mlredact.pipeline.requirements import required_model_ids
        from mlredact.registry.models import models_dir, resolve
        from mlredact.resources.loader import verify_resources
        from mlredact.runtime.profile import check_runtime_profile

        cfg = load_config(profile, config or ())
        rp = check_runtime_profile(cfg)
        n_resources = verify_resources()
        directory = models_dir(cfg.runtime.models_dir)
        model_ids = required_model_ids(cfg)
        for mid in model_ids:
            resolve(mid, directory)
        if cfg.ocr.engine_b.enabled and binary_version(cfg.ocr.engine_b.binary) is None:
            raise EnvironmentErrorMl(ReasonCode.MODEL_MISSING, engine_b_binary=True)
    except MlredactError as exc:
        _fail(exc)
        return
    report = {
        "mlredact_version": __version__,
        "profile": profile,
        "config_hash": config_hash(cfg),
        "runtime_profile": rp.profile_id,
        "libraries": rp.libraries,
        "resources_verified": n_resources,
        "models_verified": model_ids,
    }
    sys.stdout.write(canonical_json_pretty(report).decode("utf-8"))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind (keep 127.0.0.1 behind a TLS proxy or tunnel)")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option(help="Port")] = 8100,
    log_level: Annotated[str, typer.Option(help="Log level")] = "WARNING",
) -> None:
    """Run the HTTP API for the web frontend (settings: MLREDACT_API_* environment variables)."""
    install_excepthook()
    configure_logging(log_level)
    try:
        import uvicorn

        from mlredact.api.app import create_app
        from mlredact.api.settings import ApiSettings

        api = create_app(ApiSettings.from_env())
    except MlredactError as exc:
        _fail(exc)
        return
    # No access log: request paths carry signed link tokens.  Uvicorn's own records go through the
    # PII-safe logging configured above.
    uvicorn.run(api, host=host, port=port, access_log=False, log_config=None, server_header=False)


@app.command()
def profiles() -> None:
    """List built-in policy profiles."""
    for p in builtin_profiles():
        typer.echo(p)


@keys_app.command("evaluator")
def keys_evaluator(
    out: Annotated[Path, typer.Option("--out", "-o", help="Directory for evaluator.key / evaluator.pub")] = Path(),
) -> None:
    """Create an evaluator keypair for sealed sensitive manifests (keep evaluator.key secret)."""
    from mlredact.manifest.sensitive import keypair, recipient_id

    private, public = keypair()
    out.mkdir(parents=True, exist_ok=True)
    key_path = out / "evaluator.key"
    key_path.write_text(private.hex() + "\n", encoding="ascii")
    with contextlib.suppress(OSError):
        key_path.chmod(0o600)
    (out / "evaluator.pub").write_text(public.hex() + "\n", encoding="ascii")
    typer.echo(f"recipient {recipient_id(public)}: public key {out / 'evaluator.pub'}, private key {key_path}")


@sensitive_app.command("open")
def sensitive_open(
    sealed: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="sensitive-manifest.sealed.json")],
    key: Annotated[Path, typer.Option("--key", help="evaluator.key")],
) -> None:
    """Decrypt a sealed sensitive manifest to stdout (evaluator side; contains original PII)."""
    from mlredact.manifest.sensitive import open_sealed

    document = open_sealed(sealed.read_bytes(), bytes.fromhex(key.read_text("ascii").strip()))
    sys.stdout.write(canonical_json_pretty(document).decode("utf-8"))


@models_app.command("fetch")
def models_fetch(
    model_ids: Annotated[list[str] | None, typer.Argument(help="Model ids (default: all registered)")] = None,
    directory: Annotated[str, typer.Option(help="Models directory")] = ".models",
    profile: Annotated[str | None, typer.Option(help="Fetch only the models this profile needs (e.g. client)")] = None,
) -> None:
    """Download and verify model files (never used by processing code)."""
    from mlredact.pipeline.requirements import required_model_ids
    from mlredact.registry.models import fetch, load_registry, models_dir

    ids = required_model_ids(load_config(profile)) if profile is not None else model_ids or sorted(load_registry())
    for mid in ids:
        fetch(mid, models_dir(directory))
        typer.echo(f"ok {mid}")


@models_app.command("verify")
def models_verify(directory: Annotated[str, typer.Option(help="Models directory")] = ".models") -> None:
    """Verify every registered model file against its pinned SHA-256."""
    from mlredact.registry.models import load_registry, models_dir, resolve

    failures = 0
    for mid in sorted(load_registry()):
        try:
            resolve(mid, models_dir(directory))
            typer.echo(f"ok      {mid}")
        except MlredactError as exc:
            failures += 1
            typer.echo(f"FAILED  {mid}: {exc.code.value}")
    raise typer.Exit(1 if failures else 0)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
