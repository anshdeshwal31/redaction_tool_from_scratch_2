"""Document bytes in shared memory: the creating process owns the segment, and a worker that reads it changes nothing
of that (fictional bytes only)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT = textwrap.dedent(
    """
    import concurrent.futures
    import multiprocessing as mp
    import os

    from mlredact.pipeline.shared import SharedBlob, read_blob

    if __name__ == "__main__":
        with SharedBlob(b"fictional bytes") as blob:
            with concurrent.futures.ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
                assert pool.submit(read_blob, blob.ref).result() == b"fictional bytes"
            name = blob.ref.name.lstrip("/")
        print("unlinked", not os.path.exists("/dev/shm/" + name))
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shared memory and its resource tracker")
def test_a_worker_read_leaves_the_creators_registration_alone(tmp_path: Path) -> None:
    """The worker shares the creator's resource tracker: a read that removed the creator's registration made every
    close print a KeyError, and left the segment behind when the creator died before closing it."""
    script = tmp_path / "blob.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert "unlinked True" in done.stdout
    assert "KeyError" not in done.stderr
