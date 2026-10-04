"""Command-line interface.

Output files have fixed names (``redacted.pdf``, ``manifest.json``, ``run.json``,
``quarantine.json``): input filenames often contain names and are never echoed or reused.
Exit codes: 0 released, 2 quarantined, 1 usage/environment error.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
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


def _fail(exc: BaseException) -> None:
    info = sanitize(exc)
    typer.echo(f"error: {info.code.value} ({info.error_type} at {info.where})", err=True)
    raise typer.Exit(1)


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
    profile: Annotated[str, typer.Option(help="Policy profile")] = "broad",
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
            result = redactor.run(data, seeds=seed_list, scope_id=scope_id, sensitive_to=recipient)
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
    _write(
        out / "quarantine.json",
        canonical_json_pretty(
            {
                "status": "quarantined",
                "reasons": result.manifest["reasons"],
                "input_sha256": result.manifest["job"]["input_sha256"],
            }
        ),
    )
    typer.echo(f"quarantined: {', '.join(result.manifest['reasons']) or 'verification failed'}", err=True)
    raise typer.Exit(2)


@app.command()
def selfcheck(
    profile: Annotated[str, typer.Option()] = "broad",
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
    profile: Annotated[str | None, typer.Option(help="Fetch only the models this profile needs (e.g. broad)")] = None,
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
