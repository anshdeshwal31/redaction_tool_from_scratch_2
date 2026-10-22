"""Process environment of the benchmark adapter: version string, configuration files, models folder.

The benchmark starts the adapter from its own run folder, so nothing here may assume the backend
folder is the current directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import mlredact
from mlredact import __version__
from mlredact.config.schema import AppConfig

CONFIG_ENV = "MLREDACT_CONFIG"


def backend_root() -> Path:
    """The backend folder of a source checkout (the package lives in ``src/mlredact`` below it)."""
    return Path(mlredact.__file__).resolve().parents[2]


def candidate_version(root: Path | None = None) -> str:
    """``<version>+g<short commit>``, with ``.dirty`` when the backend folder has uncommitted changes;
    just ``<version>`` without git or outside a source checkout."""
    root = root if root is not None else backend_root()
    git = shutil.which("git")
    if git is None or not (root / "pyproject.toml").is_file():
        return __version__
    try:
        head = subprocess.run(  # noqa: S603 - fixed binary and arguments, no shell
            [git, "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        status = subprocess.run(  # noqa: S603 - fixed binary and arguments, no shell
            [git, "-C", str(root), "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return __version__
    commit = head.stdout.strip()
    if head.returncode != 0 or not commit:
        return __version__
    # A status that cannot be read is not evidence of a clean tree.
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return f"{__version__}+g{commit}{'.dirty' if dirty else ''}"


def env_config_files() -> list[Path]:
    """Deployment YAMLs from ``MLREDACT_CONFIG`` (``os.pathsep``-separated); they apply before ``--config``."""
    return [Path(p).expanduser() for p in os.environ.get(CONFIG_ENV, "").split(os.pathsep) if p.strip()]


def models_dir_fallback(configured: str, root: Path | None = None) -> Path | None:
    """The backend's own ``.models`` folder, when ``MLREDACT_MODELS_DIR`` is unset and the configured
    folder does not exist from the current directory (the benchmark's run folder); otherwise None."""
    if "MLREDACT_MODELS_DIR" in os.environ or Path(configured).is_dir():
        return None
    fallback = (root if root is not None else backend_root()) / ".models"
    return fallback if fallback.is_dir() else None


def surrogate_key_source(config: AppConfig) -> str | None:
    """Which surrogate key a run uses, in ``load_master_key``'s order: ``env``, ``file`` or ``development``
    (None when the render style draws no surrogates)."""
    if config.redaction.style != "surrogate":
        return None
    if os.environ.get("MLREDACT_SURROGATE_KEY"):
        return "env"
    if config.surrogate.key_file:
        return "file"
    return "development"
