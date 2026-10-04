"""Process pool: spawned workers, deterministic result order, fail-closed timeouts/crashes."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from typing import Any, TypeVar

from mlredact.config.schema import AppConfig
from mlredact.core.errors import MlredactError, ProcessingError, ReasonCode
from mlredact.pipeline.workers import init_worker
from mlredact.runtime.determinism import determinism_env

R = TypeVar("R")


class WorkerPool:
    """Spawned worker processes.  A crash or timeout kills the pool (fail-closed for that job) and the
    next call transparently starts a fresh one, so one hostile document cannot disable the service."""

    def __init__(self, config: AppConfig, workers: int | None = None, *, log_level: str = "WARNING") -> None:
        self._config = config
        self.workers = workers if workers is not None else config.runtime.workers
        self._log_level = log_level
        self._executor: ProcessPoolExecutor | None = None
        self.restarts = 0
        self._start()

    def _start(self) -> ProcessPoolExecutor:
        # Children inherit this environment at spawn time (thread limits, offline flags, hash seed).
        os.environ.update(determinism_env(self._config.runtime.threads))
        strict = os.environ.get("MLREDACT_LOG_STRICT", "0") == "1"
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=init_worker,
            initargs=(self._config.model_dump_json(), self._log_level, strict),
        )
        return self._executor

    def call(self, fn: Callable[..., R], *args: Any) -> R:
        return self.map(fn, [args])[0]

    def map(self, fn: Callable[..., R], arg_list: Sequence[tuple[Any, ...]]) -> list[R]:
        """Run ``fn(*args)`` for every args tuple; results are returned in input order."""
        timeout = self._config.runtime.page_timeout_s
        executor = self._executor
        if executor is None:
            executor = self._start()
            self.restarts += 1
        try:
            futures: list[Future[R]] = [executor.submit(fn, *args) for args in arg_list]
        except BrokenProcessPool:
            self.terminate()
            raise ProcessingError(ReasonCode.WORKER_CRASHED) from None
        results: list[R] = []
        try:
            for f in futures:
                results.append(f.result(timeout=timeout))
        except FutureTimeout:
            self.terminate()
            raise ProcessingError(ReasonCode.WORKER_TIMEOUT) from None
        except BrokenProcessPool:
            self.terminate()
            raise ProcessingError(ReasonCode.WORKER_CRASHED) from None
        except MlredactError as exc:
            for f in futures:
                f.cancel()
            raise type(exc)(exc.code, **exc.params) from None  # drop the remote traceback chain
        except Exception:
            for f in futures:
                f.cancel()
            raise ProcessingError(ReasonCode.INTERNAL) from None
        return results

    def terminate(self) -> None:
        executor, self._executor = self._executor, None
        if executor is None:
            return
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            with contextlib.suppress(Exception):
                p.kill()

    def close(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
