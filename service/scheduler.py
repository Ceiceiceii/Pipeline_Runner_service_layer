"""Cost-aware GPU scheduling: mechanism (queue + worker-runners) and policy.

Mechanism (GpuScheduler) and policy (WarmPoolPolicy implementations) are kept
separate, mirroring the kit's own split, so the harness can replay the same
workload under different policies and compare dollars.

Design decisions defended in DESIGN.md:
- one FIFO work queue; the admission bound is owned HERE (try_admit), counted
  in admitted-but-unfinished GPU items so jobs still in their CPU prelude and
  cancelled tombstones are accounted for — JobService only relays the verdict;
- long-lived worker-runner tasks that acquire a worker once, warm it once, and
  drain the queue — coalescing/batching by construction;
- transient retries run *on the held worker* (it bills while idle anyway;
  requeueing would send the retry to the back of a multi-minute line);
- scale-up targets a bounded backlog drain time; scale-down is a lazy idle TTL
  set at the cold-start break-even (the ski-rental answer).
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from pipeline_kit.clock import Clock
from pipeline_kit.errors import PoolExhaustedError, StepError
from pipeline_kit.gpu import GpuPool, Worker, WorkerState
from service.logs import log_event
from service.metrics import MetricsRegistry
from service.models import JobCancelledError
from service.retry import AttemptObserver, RetryPolicy, run_with_retries

_EWMA_ALPHA = 0.2


@dataclass
class WorkItem:
    """One GPU step (with its retry budget) waiting for a worker."""

    job_id: str
    step_name: str
    attempt_factory: Callable[[int], Awaitable[BaseModel]]
    next_attempt: Callable[[], int]
    observer: AttemptObserver
    enqueued_at: float
    future: asyncio.Future[BaseModel] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    cancelled: bool = False
    started: bool = False  # first attempt has begun on a worker


class WarmPoolPolicy(Protocol):
    """The autoscaling brain: how many workers should be warm right now."""

    idle_ttl_s: float
    min_warm: int

    def decide(
        self, *, queue_depth: int, warm_count: int, busy_count: int,
        avg_service_s: float, now: float,
    ) -> int:
        """Return the target number of warm workers (uncapped; scheduler clamps)."""
        ...


@dataclass
class AdaptivePolicy:
    """Reactive drain-time targeting up, lazy break-even TTL down.

    Scale up so the projected backlog drain time stays under
    ``target_drain_s``; scale down only after a worker has been idle for
    ``idle_ttl_s`` (default ≈ E[cold start] + margin — the ski-rental
    break-even, so the pool stays warm through quiet gaps shorter than a
    cold start and reaches zero only when demand is truly gone).
    """

    target_drain_s: float = 60.0
    idle_ttl_s: float = 60.0
    min_warm: int = 0

    def decide(
        self, *, queue_depth: int, warm_count: int, busy_count: int,
        avg_service_s: float, now: float,
    ) -> int:
        needed = math.ceil(queue_depth * avg_service_s / self.target_drain_s)
        # Never retire a worker that is mid-job, and respect the warm floor.
        return max(needed, busy_count, self.min_warm)


@dataclass
class AlwaysWarmN:
    """Hold N workers warm forever: the latency-optimal, cost-ceiling baseline."""

    n: int = 4
    idle_ttl_s: float = math.inf
    min_warm: int = 0

    def __post_init__(self) -> None:
        self.min_warm = self.n

    def decide(
        self, *, queue_depth: int, warm_count: int, busy_count: int,
        avg_service_s: float, now: float,
    ) -> int:
        return self.n


def eager_scale_to_zero(ttl_s: float = 5.0) -> AdaptivePolicy:
    """Build the cost-naive baseline: cools almost immediately when idle.

    Under gaps shorter than a cold start this thrashes: it re-pays ~45s of
    billed warm-up (plus user-visible latency) that a lazier TTL would avoid.
    """
    return AdaptivePolicy(idle_ttl_s=ttl_s)


class GpuScheduler:
    """Bounded FIFO queue drained by policy-sized, long-lived worker-runners."""

    def __init__(
        self,
        *,
        pool: GpuPool,
        clock: Clock,
        policy: WarmPoolPolicy,
        retry_policy: RetryPolicy,
        metrics: MetricsRegistry,
        max_backlog: int,
        nominal_service_s: float,
        expected_cold_start_s: float,
        rng_seed: int = 0,
    ) -> None:
        self._pool = pool
        self._clock = clock
        self._policy = policy
        self._retry = retry_policy
        self._metrics = metrics
        self.max_backlog = max_backlog
        self._avg_service_s = nominal_service_s  # EWMA, seeded from the registry
        self._expected_cold_start_s = expected_cold_start_s
        self._rng = random.Random(rng_seed)
        self._queue: deque[WorkItem] = deque()
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._runners: set[asyncio.Task[None]] = set()
        # Spawn-accurate runner count: incremented synchronously when a runner
        # task is created, decremented in its finally. The task set above lags
        # (done-callbacks fire via call_soon), so both the min_warm floor and
        # _reconcile size against this counter to stay race-free.
        self._runner_slots = 0
        # Admission accounting, in GPU work items: every admitted job reserves
        # its chain's GPU-item count up front (so jobs still in their CPU
        # prelude hold their slots), each item's resolution releases one, and
        # finalize_job releases whatever the job never got to enqueue.
        self._admitted_backlog = 0
        self._admitted_by_job: dict[str, int] = {}
        self._started = False

    # ---------------------------------------------------------------- public

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def admitted_backlog(self) -> int:
        """GPU items admitted but not yet finished (queued, in CPU prelude, or running)."""
        return self._admitted_backlog

    @property
    def started(self) -> bool:
        return self._started

    @property
    def warm_target_now(self) -> int:
        return min(self._current_target(), self._pool.capacity)

    @property
    def avg_service_s(self) -> float:
        return self._avg_service_s

    def start(self) -> None:
        self._started = True
        self._reconcile()

    def try_admit(self, job_id: str, gpu_items: int) -> bool:
        """Reserve backlog capacity for a job's GPU items, or refuse.

        The bound is counted in admitted-but-unfinished items, so a burst of
        jobs still in their CPU prelude cannot slip past a momentarily empty
        queue, and cancelled work frees its reservation immediately.
        """
        if gpu_items <= 0:
            return True
        if self._admitted_backlog + gpu_items > self.max_backlog:
            return False
        self._admitted_backlog += gpu_items
        self._admitted_by_job[job_id] = (
            self._admitted_by_job.get(job_id, 0) + gpu_items
        )
        return True

    def finalize_job(self, job_id: str) -> None:
        """Release whatever admitted capacity the job never turned into items."""
        remaining = self._admitted_by_job.pop(job_id, 0)
        if remaining:
            self._release_backlog(remaining)

    async def stop(self) -> None:
        """Cancel runners (their finally-blocks cool + release their workers)."""
        self._started = False
        for task in list(self._runners):
            task.cancel()
        await asyncio.gather(*self._runners, return_exceptions=True)
        for item in self._queue:
            if not item.future.done():
                item.future.set_exception(
                    JobCancelledError("scheduler stopped before this step ran")
                )
        self._queue.clear()

    def estimate_wait_s(self, extra_items: int = 0) -> float:
        """Projected wait for a newly admitted item, used for retry_after hints.

        Sized from the admitted backlog (not just the visible queue), so jobs
        still in their CPU prelude count toward the projection.
        """
        depth = self._admitted_backlog + extra_items
        warm = max(self._runner_slots, 1)
        estimate = depth * self._avg_service_s / warm
        if self._runner_slots == 0:
            estimate += self._expected_cold_start_s
        return estimate

    def make_item(
        self,
        *,
        job_id: str,
        step_name: str,
        attempt_factory: Callable[[int], Awaitable[BaseModel]],
        next_attempt: Callable[[], int],
        observer: AttemptObserver,
    ) -> WorkItem:
        """Create, enqueue, and return a work item (future resolves on completion).

        Never rejects: the backlog bound was already reserved at *job*
        admission (try_admit) — turning away a mid-chain step of an admitted
        job would waste the GPU seconds already spent. The item consumes one
        unit of its job's reservation and releases it when its future resolves
        (success, failure, cancellation, or shutdown alike).
        """
        item = WorkItem(
            job_id=job_id,
            step_name=step_name,
            attempt_factory=attempt_factory,
            next_attempt=next_attempt,
            observer=observer,
            enqueued_at=self._clock.monotonic(),
        )
        # Move one unit of the job's reservation onto the item itself.
        job_remaining = self._admitted_by_job.get(job_id, 0)
        if job_remaining > 1:
            self._admitted_by_job[job_id] = job_remaining - 1
        else:
            self._admitted_by_job.pop(job_id, None)
        item.future.add_done_callback(lambda _f: self._release_backlog(1))
        self._queue.append(item)
        self._metrics.inc("gpu_items_enqueued")
        if self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
        self._reconcile()
        return item

    # -------------------------------------------------------------- internals

    def _release_backlog(self, items: int) -> None:
        self._admitted_backlog = max(0, self._admitted_backlog - items)

    def _current_target(self) -> int:
        return self._policy.decide(
            queue_depth=len(self._queue),
            warm_count=len(self._runners),
            busy_count=self._pool.busy_count,
            avg_service_s=self._avg_service_s,
            now=self._clock.monotonic(),
        )

    def _reconcile(self) -> None:
        """Spawn runner tasks up to the policy target.

        Never tears down: scale-down is each runner's own lazy TTL decision.
        """
        if not self._started:
            return
        target = min(self._current_target(), self._pool.capacity)
        while self._runner_slots < target:
            self._runner_slots += 1
            task = asyncio.create_task(self._runner_loop())
            self._runners.add(task)
            task.add_done_callback(self._on_runner_done)

    def _on_runner_done(self, task: asyncio.Task[None]) -> None:
        self._runners.discard(task)
        # An item enqueued between this runner's retirement decision and this
        # callback saw a stale runner count; re-check so it can't be stranded.
        self._reconcile()

    def _pop_live(self) -> WorkItem | None:
        while self._queue:
            item = self._queue.popleft()
            if item.cancelled or item.future.done():
                if not item.future.done():
                    item.future.set_exception(
                        JobCancelledError("cancelled while queued")
                    )
                continue
            return item
        return None

    async def _runner_loop(self) -> None:
        """One warm interval: acquire once, warm once, drain until idle-TTL."""
        try:
            try:
                worker = self._pool.acquire()
            except PoolExhaustedError:
                return  # every slot is leased; the queue drains through them
            try:
                if worker.state is WorkerState.COLD:
                    self._metrics.inc("cold_starts")
                    log_event("worker.warming", worker_id=worker.id)
                await worker.warm()
                log_event("worker.warm", worker_id=worker.id)
                while True:
                    item = await self._next_item()
                    if item is None:
                        break
                    await self._execute(item, worker)
            finally:
                worker.cool()
                self._pool.release(worker)
                log_event("worker.cooled", worker_id=worker.id)
        finally:
            # Covers the acquire-failure early return too, so the slot count
            # can never drift from reality.
            self._runner_slots -= 1
            # A late burst may have queued work while we were shutting down.
            self._reconcile()

    async def _next_item(self) -> WorkItem | None:
        """Next live item, or None once idle for the TTL and allowed to retire."""
        deadline = self._clock.monotonic() + self._policy.idle_ttl_s
        while True:
            item = self._pop_live()
            if item is not None:
                return item
            remaining = deadline - self._clock.monotonic()
            if remaining <= 0:
                # No await between this check and the finally-block decrement,
                # so concurrent idle runners can't all retire past the floor.
                if self._runner_slots > self._policy.min_warm:
                    return None
                # This runner is part of the warm floor: keep waiting.
                deadline = self._clock.monotonic() + self._policy.idle_ttl_s
                continue
            await self._wait_for_work(remaining)

    async def _wait_for_work(self, timeout_s: float) -> None:
        """Sleep until an item is enqueued or ``timeout_s`` passes (clock time)."""
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._waiters.append(waiter)
        waits: set[asyncio.Future[None] | asyncio.Task[None]] = {waiter}
        timer: asyncio.Task[None] | None = None
        if math.isfinite(timeout_s):
            timer = asyncio.create_task(self._clock.sleep(timeout_s))
            waits.add(timer)
        try:
            _done, pending = await asyncio.wait(
                waits, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            if timer is not None and not timer.done():
                timer.cancel()

    async def _execute(self, item: WorkItem, worker: Worker) -> None:
        """Run one item's full retry budget on the held worker."""
        item.started = True
        started = self._clock.monotonic()
        self._metrics.observe_queue_wait(started - item.enqueued_at)
        item.observer.worker_assigned(worker.id)
        log_event(
            "gpu.item_started",
            job_id=item.job_id,
            step=item.step_name,
            worker_id=worker.id,
            queue_wait_s=round(started - item.enqueued_at, 3),
        )
        try:
            result = await run_with_retries(
                lambda attempt: worker.run(item.attempt_factory(attempt)),
                policy=self._retry,
                clock=self._clock,
                next_attempt=item.next_attempt,
                observer=item.observer,
                rng=self._rng,
                should_abort=lambda: item.cancelled,
            )
        except StepError as exc:
            if not item.future.done():
                item.future.set_exception(exc)
        except JobCancelledError as exc:
            # Cancelled between retry attempts: stop burning the budget.
            if not item.future.done():
                item.future.set_exception(exc)
        except asyncio.CancelledError:
            if not item.future.done():
                item.future.set_exception(
                    JobCancelledError("scheduler stopped mid-step")
                )
            raise
        else:
            if not item.future.done():
                item.future.set_result(result)
        finally:
            held_s = self._clock.monotonic() - started
            self._avg_service_s = (
                (1 - _EWMA_ALPHA) * self._avg_service_s + _EWMA_ALPHA * held_s
            )
            self._reconcile()
