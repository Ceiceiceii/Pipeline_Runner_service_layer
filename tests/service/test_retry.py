"""RetryPolicy backoff math and the shared attempt loop."""

from __future__ import annotations

import pytest

from pipeline_kit.clock import ManualClock
from pipeline_kit.errors import PermanentError, TransientError
from service.retry import RetryExhaustedError, RetryPolicy, run_with_retries
from tests.service.conftest import settle


def _error(cls, attempt: int):
    return cls("boom", step_name="step", input_id="input", attempt=attempt)


def test_backoff_sequence_without_jitter():
    policy = RetryPolicy(base_s=1.0, factor=2.0, cap_s=15.0, jitter=False)
    assert [policy.backoff_s(n) for n in (1, 2, 3, 4, 5)] == [1.0, 2.0, 4.0, 8.0, 15.0]


def test_backoff_jitter_bounded():
    policy = RetryPolicy(base_s=1.0, factor=2.0, cap_s=15.0, jitter=True)
    import random

    rng = random.Random(0)
    for failures in range(1, 6):
        raw = min(15.0, 2.0 ** (failures - 1))
        assert 0.0 <= policy.backoff_s(failures, rng) <= raw


async def test_success_after_transient_failures():
    clock = ManualClock()
    attempts_seen: list[int] = []

    async def flaky(attempt: int) -> str:
        attempts_seen.append(attempt)
        if len(attempts_seen) < 3:
            raise _error(TransientError, attempt)
        return "ok"

    counter = iter(range(100))
    task = run_with_retries(
        flaky,
        policy=RetryPolicy(jitter=False),
        clock=clock,
        next_attempt=lambda: next(counter),
    )
    import asyncio

    running = asyncio.ensure_future(task)
    await settle(clock, until=running.done)
    assert await running == "ok"
    assert attempts_seen == [0, 1, 2]  # fresh attempt number per try


async def test_transient_exhaustion_raises_after_max_attempts():
    clock = ManualClock()
    calls = 0

    async def always_fails(attempt: int) -> str:
        nonlocal calls
        calls += 1
        raise _error(TransientError, attempt)

    counter = iter(range(100))
    import asyncio

    running = asyncio.ensure_future(
        run_with_retries(
            always_fails,
            policy=RetryPolicy(max_attempts=4, jitter=False),
            clock=clock,
            next_attempt=lambda: next(counter),
        )
    )
    await settle(clock, until=running.done)
    with pytest.raises(RetryExhaustedError) as excinfo:
        await running
    assert calls == 4
    assert excinfo.value.attempts == 4


async def test_permanent_error_short_circuits():
    clock = ManualClock()
    calls = 0

    async def poisoned(attempt: int) -> str:
        nonlocal calls
        calls += 1
        raise _error(PermanentError, attempt)

    import asyncio

    running = asyncio.ensure_future(
        run_with_retries(
            poisoned,
            policy=RetryPolicy(max_attempts=4, jitter=False),
            clock=clock,
            next_attempt=lambda: 0,
        )
    )
    await settle(clock, until=running.done)
    with pytest.raises(PermanentError):
        await running
    assert calls == 1
