"""Generic chain executor, driven entirely by the STEPS registry.

CPU steps run inline (they're free); GPU steps become work items on the
scheduler. Outputs accumulate in a type-keyed context, mirroring how the
registry declares inputs, so no per-step code exists anywhere in the runner.
"""

from __future__ import annotations

import asyncio
import random
from collections import OrderedDict
from collections.abc import Awaitable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from pipeline_kit.clock import Clock
from pipeline_kit.config import KitSettings
from pipeline_kit.errors import PermanentError, StepError
from pipeline_kit.pipelines import STEPS
from pipeline_kit.schemas import Image
from service.logs import log_event
from service.models import (
    ERROR_KINDS,
    AttemptSpan,
    ErrorKind,
    Job,
    JobCancelledError,
    JobError,
    JobState,
    StepRun,
    StepState,
)
from service.resources import ResourceClass, resource_class
from service.retry import RetryExhaustedError, RetryPolicy, run_with_retries

if TYPE_CHECKING:
    from service.metrics import MetricsRegistry
    from service.scheduler import GpuScheduler, WorkItem

MemoKey = tuple[str, tuple[str, ...]]


def _artifact_id(model: BaseModel) -> str:
    """Content-addressed id every kit output carries (not part of BaseModel)."""
    return str(model.id)  # type: ignore[attr-defined]


class AttemptLedger:
    """Globally unique attempt numbers per (step, input).

    The kit draws failure as a pure function of ``(step, input, attempt)``, so
    every attempt — including one made by a resubmission of a failed job —
    must use a fresh number to get an independent draw.
    """

    def __init__(self) -> None:
        self._used: dict[tuple[str, str], int] = {}

    def next_attempt(self, step_name: str, input_id: str) -> int:
        key = (step_name, input_id)
        attempt = self._used.get(key, 0)
        self._used[key] = attempt + 1
        return attempt


class StepRunObserver:
    """Writes attempt spans into a StepRun as the retry loop reports them."""

    def __init__(self, step_run: StepRun, metrics: MetricsRegistry) -> None:
        self._step_run = step_run
        self._metrics = metrics
        self._worker_id: int | None = None

    def worker_assigned(self, worker_id: int) -> None:
        self._worker_id = worker_id

    def attempt_started(self, attempt: int, t: float) -> None:
        self._step_run.state = StepState.RUNNING
        self._step_run.attempts.append(
            AttemptSpan(attempt=attempt, t_start=t, worker_id=self._worker_id)
        )
        self._metrics.inc("step_attempts")

    def attempt_finished(self, attempt: int, t: float, error: StepError | None) -> None:
        span = self._step_run.attempts[-1]
        span.t_end = t
        if error is None:
            span.outcome = "succeeded"
            return
        self._metrics.inc("step_attempt_failures")
        if isinstance(error, PermanentError):
            span.outcome = "permanent_error"
        else:
            span.outcome = "transient_error"
            self._step_run.state = StepState.RETRYING
            self._metrics.inc("step_retries")
        log_event(
            "step.attempt_failed",
            step=self._step_run.step_name,
            attempt=attempt,
            error=str(error),
        )


class ChainRunner:
    """Executes one job's chain: CPU inline, GPU via the scheduler."""

    def __init__(
        self,
        *,
        settings: KitSettings,
        clock: Clock,
        scheduler: GpuScheduler,
        retry_policy: RetryPolicy,
        metrics: MetricsRegistry,
        memo: OrderedDict[MemoKey, BaseModel] | None = None,
        ledger: AttemptLedger | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._scheduler = scheduler
        self._retry = retry_policy
        self._metrics = metrics
        # LRU-bounded: entries hold full artifact models, and a long-running
        # process must not grow without limit. Durable memoization belongs
        # behind the store seam (declared skip in DESIGN.md).
        self._memo: OrderedDict[MemoKey, BaseModel] = OrderedDict(memo or {})
        self._memo_cap = 4096
        self._ledger = ledger or AttemptLedger()
        self._rng = random.Random(settings.seed)  # backoff jitter, deterministic
        # job_id -> currently awaited GPU WorkItem (used by JobService.cancel)
        self._active_items: dict[str, WorkItem] = {}

    async def run(self, job: Job) -> None:
        """Drive the job to a terminal state.

        Never leaves the job non-terminal: step failures and cancellations
        (including task cancellation from service shutdown) all finalize the
        job before returning or re-raising.
        """
        job.state = JobState.RUNNING
        job.started_at = self._clock.monotonic()
        context: dict[type[BaseModel], BaseModel] = {Image: job.image}
        for step_run in job.steps:
            if job.cancel_requested:
                self._finish_cancelled(job, step_run)
                return
            try:
                output = await self._run_step(job, step_run, context)
            except StepError as exc:
                self._finish_failed(job, step_run, exc)
                return
            except JobCancelledError:
                self._finish_cancelled(job, step_run)
                return
            except asyncio.CancelledError:
                # Service shutdown cancelled this task mid-step: the job must
                # still reach a terminal state before the task dies.
                self._finish_cancelled(job, step_run)
                raise
            context[type(output)] = output
            step_run.output_id = _artifact_id(output)
            job.result_ids[step_run.step_name] = _artifact_id(output)
        job.state = JobState.SUCCEEDED
        job.finished_at = self._clock.monotonic()
        log_event("job.succeeded", job_id=job.job_id, result_id=job.final_result_id)

    async def _run_step(
        self,
        job: Job,
        step_run: StepRun,
        context: dict[type[BaseModel], BaseModel],
    ) -> BaseModel:
        info = STEPS[step_run.step_name]
        inputs = [context[model] for model in info.input_models]
        input_ids: tuple[str, ...] = tuple(item.id for item in inputs)  # type: ignore[attr-defined]

        memo_key: MemoKey = (info.name, input_ids)
        cached = self._memo.get(memo_key)
        if cached is not None:
            self._memo.move_to_end(memo_key)
            step_run.state = StepState.SUCCEEDED
            step_run.memoized = True
            step_run.output_id = cached.id  # type: ignore[attr-defined]
            self._metrics.inc("steps_memoized")
            log_event("step.memoized", job_id=job.job_id, step=info.name)
            return cached

        observer = StepRunObserver(step_run, self._metrics)
        ledger_id = input_ids[0] if input_ids else job.image.id

        def next_attempt() -> int:
            return self._ledger.next_attempt(info.name, ledger_id)

        def make_attempt(attempt: int) -> Awaitable[BaseModel]:
            return info.fn(
                *inputs, settings=self._settings, clock=self._clock, attempt=attempt
            )

        if resource_class(info.name) is ResourceClass.GPU:
            item = self._scheduler.make_item(
                job_id=job.job_id,
                step_name=info.name,
                attempt_factory=make_attempt,
                next_attempt=next_attempt,
                observer=observer,
            )
            step_run.enqueued_at = item.enqueued_at
            self._register_item(job, item)
            try:
                output = await item.future
            finally:
                self._register_item(job, None)
        else:
            step_run.state = StepState.RUNNING
            output = await run_with_retries(
                make_attempt,
                policy=self._retry,
                clock=self._clock,
                next_attempt=next_attempt,
                observer=observer,
                rng=self._rng,
                should_abort=lambda: job.cancel_requested,
            )

        step_run.state = StepState.SUCCEEDED
        self._memo[memo_key] = output
        while len(self._memo) > self._memo_cap:
            self._memo.popitem(last=False)
        return output

    def _register_item(self, job: Job, item: WorkItem | None) -> None:
        """Track the job's queued/in-flight GPU item so cancel can reach it."""
        self._active_items.pop(job.job_id, None)
        if item is not None:
            self._active_items[job.job_id] = item

    def active_item(self, job_id: str) -> WorkItem | None:
        return self._active_items.get(job_id)

    def _finish_failed(self, job: Job, step_run: StepRun, exc: StepError) -> None:
        # run_with_retries only lets RetryExhaustedError and PermanentError
        # escape; anything else here is a programming error worth surfacing.
        kind: ErrorKind = (
            "transient_exhausted" if isinstance(exc, RetryExhaustedError) else "permanent"
        )
        error = JobError(
            step_name=step_run.step_name,
            attempt=exc.attempt,
            kind=kind,
            retryable=ERROR_KINDS[kind].retryable,
            message=str(exc),
        )
        step_run.state = StepState.FAILED
        step_run.error = error
        job.state = JobState.FAILED
        job.error = error
        job.finished_at = self._clock.monotonic()
        log_event(
            "job.failed",
            job_id=job.job_id,
            step=step_run.step_name,
            kind=kind,
            attempt=exc.attempt,
        )

    def _finish_cancelled(self, job: Job, step_run: StepRun) -> None:
        # Finalize the interrupted step too: a terminal job must never carry
        # a RUNNING/RETRYING step or an open attempt span.
        if step_run.state in (StepState.RUNNING, StepState.RETRYING):
            step_run.state = StepState.CANCELLED
        if step_run.attempts and step_run.attempts[-1].t_end is None:
            step_run.attempts[-1].t_end = self._clock.monotonic()
        job.state = JobState.CANCELLED
        job.error = JobError(
            step_name=step_run.step_name,
            attempt=len(step_run.attempts),
            kind="cancelled",
            retryable=ERROR_KINDS["cancelled"].retryable,
            message="job cancelled before this step completed",
        )
        job.finished_at = self._clock.monotonic()
        log_event("job.cancelled", job_id=job.job_id, step=step_run.step_name)
