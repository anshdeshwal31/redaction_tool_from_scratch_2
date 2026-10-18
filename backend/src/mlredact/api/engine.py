"""Running the CLI as a child process: spawning, the stderr reason-code filter, and killing a whole
process tree (the CLI's page workers included)."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import IO

from mlredact.api.settings import ApiSettings

# The CLI prints content-free diagnostics: ``error: <CODE> (<type> at <where>)`` on exit 1.  Only
# that code is kept; everything else on stderr (third-party log lines included) is discarded
# unread, so nothing document-derived reaches a job record or a response.
_ERROR_LINE = re.compile(r"^error: ((?:E|V\d)_[A-Z0-9_]+)\b")
_MAX_LINE = 16_384
_MAX_STDOUT = 1_000_000
_IS_WINDOWS = os.name == "nt"
# How long a SIGINT gets before the whole group is sent SIGKILL.
KILL_GRACE_S = 10.0


def spawn(settings: ApiSettings, args: Sequence[str], *, capture_stdout: bool = False) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 - fixed command from settings, no shell
        [*settings.engine, *args],
        cwd=settings.engine_cwd,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        # Own process group on POSIX, so the whole tree can be signalled.
        start_new_session=not _IS_WINDOWS,
    )


def watch_error_code(stream: IO[bytes]) -> Callable[[], str | None]:
    """Read ``stream`` on a thread; the returned getter waits for EOF and gives the first code."""
    found: list[str] = []

    def run() -> None:
        skipping = False
        while chunk := stream.readline(_MAX_LINE):
            complete = chunk.endswith(b"\n")
            overlong = not complete and len(chunk) >= _MAX_LINE
            # Only the CLI's own diagnostic counts: at the start of a line, never indented, and never
            # the tail of an overlong line.
            if not skipping and not overlong and not found:
                match = _ERROR_LINE.match(chunk.decode("utf-8", "replace").rstrip())
                if match:
                    found.append(match.group(1))
            skipping = overlong or (skipping and not complete)
        stream.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def result() -> str | None:
        thread.join(timeout=5)
        return found[0] if found else None

    return result


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Stop a child and everything it started.  POSIX gets SIGINT first so Python releases its shared
    memory, then SIGKILL after a grace period."""
    if _IS_WINDOWS:
        if proc.poll() is not None:
            return
        subprocess.run(  # noqa: S603 - fixed system binary
            ["taskkill", "/pid", str(proc.pid), "/T", "/F"],  # noqa: S607
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    killpg = getattr(os, "killpg")  # noqa: B009 - POSIX only; keeps mypy quiet on Windows
    # The group, not the leader: page workers can outlive a CLI that has already exited, and the
    # group id stays theirs (it is not reused) for as long as any of them is alive.
    try:
        killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return  # nobody left in the group

    def force() -> None:
        with contextlib.suppress(ProcessLookupError):
            killpg(proc.pid, getattr(signal, "SIGKILL"))  # noqa: B009

    timer = threading.Timer(KILL_GRACE_S, force)
    timer.daemon = True
    timer.start()


@dataclass(frozen=True)
class EngineResult:
    code: int | None
    stdout: str
    error_code: str | None
    spawn_failed: bool
    timed_out: bool


def run_short(settings: ApiSettings, args: Sequence[str], timeout_s: float) -> EngineResult:
    """Run a short CLI command (selfcheck) and collect its stdout."""
    try:
        proc = spawn(settings, args, capture_stdout=True)
    except OSError:
        return EngineResult(None, "", None, spawn_failed=True, timed_out=False)
    assert proc.stderr is not None and proc.stdout is not None
    error_code = watch_error_code(proc.stderr)
    out: list[bytes] = []

    def drain(stream: IO[bytes]) -> None:
        # Keep the first _MAX_STDOUT bytes but read to EOF: a child blocked on a full pipe never exits.
        kept = 0
        while chunk := stream.read(65_536):
            if kept < _MAX_STDOUT:
                out.append(chunk[: _MAX_STDOUT - kept])
                kept += len(out[-1])

    reader = threading.Thread(target=drain, args=(proc.stdout,), daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc)
        proc.wait()
    reader.join(timeout=5)
    stdout = b"".join(out).decode("utf-8", "replace")
    return EngineResult(proc.returncode, stdout, error_code(), spawn_failed=False, timed_out=timed_out)
