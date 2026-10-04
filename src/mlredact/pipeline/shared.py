"""Document bytes shared with worker processes through anonymous shared memory (RAM only)."""

from __future__ import annotations

import contextlib
import hashlib
import sys
from dataclasses import dataclass
from multiprocessing import shared_memory


@dataclass(frozen=True, slots=True)
class BlobRef:
    name: str
    size: int
    key: str  # content hash prefix, used as a per-worker cache key


class SharedBlob:
    def __init__(self, data: bytes) -> None:
        self._shm = shared_memory.SharedMemory(create=True, size=max(1, len(data)))
        buf = self._shm.buf
        assert buf is not None
        buf[: len(data)] = data
        self.ref = BlobRef(self._shm.name, len(data), hashlib.sha256(data).hexdigest()[:24])

    def close(self) -> None:
        try:
            self._shm.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                self._shm.unlink()

    def __enter__(self) -> SharedBlob:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_blob(ref: BlobRef) -> bytes:
    """Copy the blob into this process.  Attaching processes must not unlink it on exit."""
    shm = shared_memory.SharedMemory(name=ref.name, create=False)
    try:
        if sys.platform != "win32":  # Python < 3.13: attachers are tracked and would unlink on exit
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        buf = shm.buf
        assert buf is not None
        return bytes(buf[: ref.size])
    finally:
        shm.close()
