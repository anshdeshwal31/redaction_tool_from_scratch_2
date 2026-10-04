"""Worker pool: errors cross the process boundary intact and a crashed pool recovers."""

from __future__ import annotations

import pickle

import pytest

from mlredact.config.loader import load_config
from mlredact.core.errors import InputRejected, MlredactError, ProcessingError, ReasonCode, VerificationFailed
from mlredact.pipeline.pool import WorkerPool
from support import task_crash, task_echo, task_reject


@pytest.mark.parametrize(
    "err",
    [
        InputRejected(ReasonCode.INPUT_ENCRYPTED),
        ProcessingError(ReasonCode.INTERNAL, page=3, retry=False),
        VerificationFailed(ReasonCode.VERIFY_TEXT_LEAK),
    ],
)
def test_errors_survive_pickling(err: MlredactError) -> None:
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is type(err) and back.code is err.code and back.params == err.params
    assert str(back) == str(err)


def test_worker_errors_keep_their_code_and_a_crash_does_not_disable_the_pool() -> None:
    with WorkerPool(load_config("broad"), workers=1) as pool:
        with pytest.raises(InputRejected) as info:
            pool.call(task_reject, 4)
        assert info.value.code is ReasonCode.INPUT_ENCRYPTED and info.value.params == {"page": 4}
        assert pool.map(task_echo, [(1,), (2,), (3,)]) == [2, 4, 6]  # still healthy after an error

        with pytest.raises(ProcessingError) as crashed:
            pool.call(task_crash)
        assert crashed.value.code is ReasonCode.WORKER_CRASHED
        assert pool.call(task_echo, 21) == 42  # a fresh pool was started
        assert pool.restarts == 1
