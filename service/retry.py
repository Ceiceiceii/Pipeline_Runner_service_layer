"""Retry policy: backoff math plus the shared attempt loop.

The same loop drives CPU steps (inline in the runner) and GPU steps (inside a
worker lease in the scheduler), so retry semantics cannot drift between the two
paths. Backoff sleeps happen on the injected clock; where the sleep happens —
holding a GPU worker vs. not — is the caller's economics decision (DESIGN.md).
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pipeline_kit.clock import Clock
from pipeline_kit.errors import PermanentError, StepError, TransientError
from service.models import JobCancelledError

T = TypeVar("T")


class RetryExhaustedError(StepError):
    """A step kept failing transiently until the attempt budget ran out."""

    def __init__(self, last: TransientError, *, attempts: int) -> None:
        super().__init__(
            f"{last.step_name} failed {attempts} attempts; giving up: {last}",
            step_name=last.step_name,
            input_id=last.input_id,
            attempt=last.attempt,
        )
        self.attempts = attempts
        self.last = last


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    At the kit's default 10% per-attempt transient failure rate, 4 attempts
    leave ~1e-4 residual failure. The cap bounds how long a retrying GPU step
    can hold its worker.
    """

    max_attempts: int = 4
    base_s: float = 1.0
    factor: float = 2.0
    cap_s: float = 15.0
    jitter: bool = True

    def backoff_s(self, failures: int, rng: random.Random | None = None) -> float:
        """Sleep before the next attempt, after ``failures`` failures (>=1)."""
        raw = min(self.cap_s, self.base_s * self.factor ** (failures - 1))
        if not self.jitter:
            return raw
        return (rng or random).uniform(0.0, raw)


class AttemptObserver(Protocol):
    """Callbacks that let the caller record attempt spans as they happen."""

    def attempt_started(self, attempt: int, t: float) -> None: ...

    def attempt_finished(
        self, attempt: int, t: float, error: StepError | None
    ) -> None: ...

    def worker_assigned(self, worker_id: int) -> None: ...


class NullObserver:
    """Observer that records nothing."""

    def attempt_started(self, attempt: int, t: float) -> None: ...

    def attempt_finished(
        self, attempt: int, t: float, error: StepError | None
    ) -> None: ...

    def worker_assigned(self, worker_id: int) -> None: ...


async def run_with_retries(
    execute_attempt: Callable[[int], Awaitable[T]],
    *,
    policy: RetryPolicy,
    clock: Clock,
    next_attempt: Callable[[], int],
    observer: AttemptObserver | None = None,
    rng: random.Random | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> T:
    """Run one step to success, retrying transient failures per ``policy``.

    ``next_attempt`` supplies globally unique attempt numbers per (step, input)
    — the kit draws failure as a pure function of ``(step, input, attempt)``,
    so fresh numbers give retries (and resubmits of failed jobs) an independent
    chance instead of deterministically repeating the same draw.

    ``should_abort`` is consulted between attempts (after a transient failure,
    before the backoff sleep): a cancelled job must not burn the rest of its
    retry budget on a billed worker. Raises ``JobCancelledError`` then.

    Raises ``PermanentError`` immediately, ``RetryExhaustedError`` after
    ``policy.max_attempts`` transient failures.
    """
    observer = observer or NullObserver()
    failures = 0
    while True:
        attempt = next_attempt()
        observer.attempt_started(attempt, clock.monotonic())
        try:
            result = await execute_attempt(attempt)
        except TransientError as exc:
            observer.attempt_finished(attempt, clock.monotonic(), exc)
            failures += 1
            if failures >= policy.max_attempts:
                raise RetryExhaustedError(exc, attempts=failures) from exc
            if should_abort is not None and should_abort():
                raise JobCancelledError(
                    "cancelled between retry attempts"
                ) from exc
            await clock.sleep(policy.backoff_s(failures, rng))
        except PermanentError as exc:
            observer.attempt_finished(attempt, clock.monotonic(), exc)
            raise
        else:
            observer.attempt_finished(attempt, clock.monotonic(), None)
            return result
