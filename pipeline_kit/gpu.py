"""A simulated, cost-metered pool of GPU workers — mechanism, never policy.

This is the substrate the exercise is built around. It faithfully models the
*physics* of a scarce GPU fleet and exposes raw cost/time signals, but it makes
**no scheduling decision for you**: it never auto-warms, never queues, never
sheds, never batches, never scales. When and how to do those things is the whole
point of the take-home.

Cost model (the rules that make the numbers trustworthy):

* A worker bills ``cost_per_second`` for the entire time it is **warm** — that
  means WARMING (paying cold start), IDLE (warm but doing nothing), and BUSY
  alike. Only a COLD worker is free.
* Cost is therefore a function of *warm wall-clock time*, owned by each worker's
  interval ledger. Running a job adds **no** cost of its own, so two items on one
  warm worker can't double-bill — coalescing work onto a warm worker is rewarded.
* A failing job still bills the warm time it occupied, and the worker is always
  returned to IDLE afterward (no stranded slots).
* Each worker is a single GPU slot: it runs one job at a time.

All times come from an injected :class:`~pipeline_kit.clock.Clock`, shared with
the workload driver so cost stays correct even under time-compressed replay.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Iterator
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel

from pipeline_kit.clock import Clock
from pipeline_kit.config import KitSettings
from pipeline_kit.determinism import rng_for
from pipeline_kit.errors import (
    PoolExhaustedError,
    WorkerBusyError,
    WorkerNotReadyError,
)

T = TypeVar("T")


class WorkerState(StrEnum):
    """Lifecycle state of a GPU worker. Everything but COLD bills."""

    COLD = "cold"
    WARMING = "warming"
    IDLE = "idle"
    BUSY = "busy"


class Worker:
    """One GPU worker: a single slot with a cold-start delay and a cost ledger."""

    def __init__(
        self,
        worker_id: int,
        *,
        cold_start_min_s: float,
        cold_start_max_s: float,
        cost_per_second: float,
        clock: Clock,
        seed: int,
    ) -> None:
        self.id = worker_id
        self.leased = False
        self._cold_start_min_s = cold_start_min_s
        self._cold_start_max_s = cold_start_max_s
        self._cost_per_second = cost_per_second
        self._clock = clock
        self._seed = seed
        self._state = WorkerState.COLD
        self._warm_started_at: float | None = None
        self._warm_seconds_closed = 0.0
        self._busy_started_at: float | None = None
        self._busy_seconds_closed = 0.0
        self._warm_count = 0

    @property
    def state(self) -> WorkerState:
        """The worker's current lifecycle state."""
        return self._state

    async def warm(self) -> None:
        """Warm a COLD worker, paying the cold-start delay before it becomes IDLE.

        Billing starts the instant warming begins — a GPU spinning up already
        costs money. No-op if the worker is not COLD, so warming is idempotent.
        """
        if self._state != WorkerState.COLD:
            return
        self._state = WorkerState.WARMING
        self._warm_started_at = self._clock.monotonic()
        cold_start = rng_for(
            self._seed, "coldstart", self.id, self._warm_count
        ).uniform(self._cold_start_min_s, self._cold_start_max_s)
        self._warm_count += 1
        await self._clock.sleep(cold_start)
        if self._state == WorkerState.WARMING:  # may have been cooled mid-warm
            self._state = WorkerState.IDLE

    def cool(self) -> None:
        """Shut the worker down to COLD, closing its warm interval and stopping cost."""
        if self._state == WorkerState.COLD:
            return
        self._close_warm_interval()
        self._close_busy_interval()
        self._state = WorkerState.COLD

    async def run(self, coro: Awaitable[T]) -> T:
        """Run one job on this warm worker.

        Requires the worker to be IDLE; raises if it is COLD/WARMING (warm it
        first) or already BUSY (one job per slot). The slot is released back to
        IDLE in a ``finally`` even if the job raises.
        """
        if self._state == WorkerState.BUSY:
            raise WorkerBusyError(f"worker {self.id} is already running a job")
        if self._state != WorkerState.IDLE:
            raise WorkerNotReadyError(
                f"worker {self.id} is {self._state.value}; warm() it before run()"
            )
        self._state = WorkerState.BUSY
        self._busy_started_at = self._clock.monotonic()
        try:
            return await coro
        finally:
            self._close_busy_interval()
            self._state = (
                WorkerState.IDLE
                if self._warm_started_at is not None
                else WorkerState.COLD
            )

    def warm_seconds(self, now: float | None = None) -> float:
        """Total seconds this worker has spent warm (closed + currently open)."""
        now = self._clock.monotonic() if now is None else now
        total = self._warm_seconds_closed
        if self._warm_started_at is not None:
            total += now - self._warm_started_at
        return total

    def busy_seconds(self, now: float | None = None) -> float:
        """Total seconds this worker has spent running jobs."""
        now = self._clock.monotonic() if now is None else now
        total = self._busy_seconds_closed
        if self._busy_started_at is not None:
            total += now - self._busy_started_at
        return total

    def cost(self, now: float | None = None) -> float:
        """Dollars billed so far: warm seconds x cost-per-second."""
        return self.warm_seconds(now) * self._cost_per_second

    def _close_warm_interval(self) -> None:
        if self._warm_started_at is not None:
            now = self._clock.monotonic()
            self._warm_seconds_closed += now - self._warm_started_at
            self._warm_started_at = None

    def _close_busy_interval(self) -> None:
        if self._busy_started_at is not None:
            now = self._clock.monotonic()
            self._busy_seconds_closed += now - self._busy_started_at
            self._busy_started_at = None


class CostReport(BaseModel):
    """A point-in-time snapshot of pool economics."""

    total_cost: float
    burn_rate_per_second: float
    warm_worker_seconds: float
    busy_worker_seconds: float
    utilization: float


class WorkerSnapshot(BaseModel):
    """A consistent read of one worker's state and cost."""

    id: int
    state: WorkerState
    leased: bool
    warm_seconds: float
    busy_seconds: float
    cost: float


class PoolSnapshot(BaseModel):
    """A consistent, immutable read of the whole pool at one instant."""

    at: float
    capacity: int
    warm_count: int
    busy_count: int
    free_count: int
    cost: CostReport
    workers: list[WorkerSnapshot]


class GpuPool:
    """A fixed-size fleet of GPU workers with a cost meter and no scheduling policy."""

    def __init__(
        self,
        *,
        max_workers: int,
        cold_start_min_s: float,
        cold_start_max_s: float,
        cost_per_second: float,
        clock: Clock,
        seed: int = 0,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._clock = clock
        self._cost_per_second = cost_per_second
        self._workers = [
            Worker(
                worker_id,
                cold_start_min_s=cold_start_min_s,
                cold_start_max_s=cold_start_max_s,
                cost_per_second=cost_per_second,
                clock=clock,
                seed=seed,
            )
            for worker_id in range(max_workers)
        ]

    @classmethod
    def from_settings(cls, settings: KitSettings, clock: Clock) -> GpuPool:
        """Build a pool from :class:`KitSettings` (the usual entry point)."""
        return cls(
            max_workers=settings.max_workers,
            cold_start_min_s=settings.cold_start_min_s,
            cold_start_max_s=settings.cold_start_max_s,
            cost_per_second=settings.cost_per_second,
            clock=clock,
            seed=settings.seed,
        )

    @property
    def capacity(self) -> int:
        """The maximum number of workers (hard cap on concurrent GPU jobs)."""
        return len(self._workers)

    @property
    def workers(self) -> tuple[Worker, ...]:
        """A read-only view of every worker."""
        return tuple(self._workers)

    @property
    def free_workers(self) -> tuple[Worker, ...]:
        """Workers not currently leased (available to acquire)."""
        return tuple(worker for worker in self._workers if not worker.leased)

    @property
    def busy_count(self) -> int:
        """How many workers are currently running a job."""
        return sum(1 for worker in self._workers if worker.state == WorkerState.BUSY)

    @property
    def warm_count(self) -> int:
        """How many workers are currently warm (and therefore billing)."""
        return sum(1 for worker in self._workers if worker.state != WorkerState.COLD)

    def acquire(self) -> Worker:
        """Lease a free worker, or raise :class:`PoolExhaustedError` if none is free.

        This never blocks or queues: backpressure is your policy to design.
        """
        for worker in self._workers:
            if not worker.leased:
                worker.leased = True
                return worker
        raise PoolExhaustedError(f"all {self.capacity} workers are leased")

    def release(self, worker: Worker) -> None:
        """Return a leased worker to the pool. Warmth is unchanged by leasing."""
        worker.leased = False

    @contextlib.contextmanager
    def lease(self) -> Iterator[Worker]:
        """Acquire a worker for the duration of a ``with`` block, releasing on exit.

        Still raises :class:`PoolExhaustedError` when the pool is full — the
        context manager handles *cleanup*, not *waiting*.
        """
        worker = self.acquire()
        try:
            yield worker
        finally:
            self.release(worker)

    def cost_report(self, now: float | None = None) -> CostReport:
        """Aggregate the raw cost/utilization signals across the fleet."""
        now = self._clock.monotonic() if now is None else now
        warm = sum(worker.warm_seconds(now) for worker in self._workers)
        busy = sum(worker.busy_seconds(now) for worker in self._workers)
        return CostReport(
            total_cost=warm * self._cost_per_second,
            burn_rate_per_second=self.warm_count * self._cost_per_second,
            warm_worker_seconds=warm,
            busy_worker_seconds=busy,
            utilization=(busy / warm) if warm > 0 else 0.0,
        )

    def snapshot(self, now: float | None = None) -> PoolSnapshot:
        """Return a consistent one-shot read of every worker plus pool economics.

        Use this in a metrics loop instead of iterating ``workers`` while
        coroutines mutate state, which can produce torn reads.
        """
        now = self._clock.monotonic() if now is None else now
        workers = [
            WorkerSnapshot(
                id=worker.id,
                state=worker.state,
                leased=worker.leased,
                warm_seconds=worker.warm_seconds(now),
                busy_seconds=worker.busy_seconds(now),
                cost=worker.cost(now),
            )
            for worker in self._workers
        ]
        return PoolSnapshot(
            at=now,
            capacity=self.capacity,
            warm_count=self.warm_count,
            busy_count=self.busy_count,
            free_count=len(self.free_workers),
            cost=self.cost_report(now),
            workers=workers,
        )
