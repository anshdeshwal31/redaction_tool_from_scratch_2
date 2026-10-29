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
    """Copy the blob into this process.  The creating process owns the blob: it unlinks it on close, and its resource
    tracker unlinks it if it dies first.  Pool workers are spawned by the creator and share that tracker, so an attach
    registers nothing new and must remove nothing either (removing the registration on attach removed the creator's
    own: a crashed job then left the document in /dev/shm, and every normal close printed a KeyError)."""
    if sys.version_info >= (3, 13):
        shm = shared_memory.SharedMemory(name=ref.name, create=False, track=False)
    else:
        shm = shared_memory.SharedMemory(name=ref.name, create=False)
    try:
        buf = shm.buf
        assert buf is not None
        return bytes(buf[: ref.size])
    finally:
        shm.close()
