"""JobService: the facade both the REST API and the agent tools drive.

Owns wiring (one settings, one clock, one pool — injected everywhere so the
dollars stay honest), the job lifecycle, admission control, idempotency, and
cancellation. Submit is synchronous: all it does is validate, dedup, admit,
record, and spawn the runner task.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from dataclasses import dataclass
from typing import Any

from pipeline_kit.clock import Clock, RealClock
from pipeline_kit.config import KitSettings
from pipeline_kit.gpu import GpuPool
from pipeline_kit.schemas import Image
from service.logs import configure_logging, log_event
from service.metrics import MetricsRegistry, build_trace, gpu_economics
from service.models import (
    Job,
    JobCancelledError,
    JobState,
    OverCapacityError,
    ServiceError,
    StepRun,
    derived_idempotency_key,
    resolve_chain,
    validate_chain,
)
from service.resources import gpu_item_count, gpu_step_durations
from service.retry import RetryPolicy
from service.runner import AttemptLedger, ChainRunner
from service.scheduler import AdaptivePolicy, GpuScheduler, WarmPoolPolicy
from service.store import InMemoryJobStore, JobStore


@dataclass(frozen=True)
class ServiceConfig:
    """Service-level knobs, each derived from a stated promise (DESIGN.md)."""

    # Admission bound in GPU work items: ~= capacity x max_acceptable_wait /
    # mean GPU seconds per item. Reserved at job admission for the whole
    # chain, so a burst still in its CPU prelude cannot slip past the bound.
    max_gpu_backlog: int = 120
    max_acceptable_wait_s: float = 300.0

    @classmethod
    def from_settings(
        cls, settings: KitSettings, max_acceptable_wait_s: float = 300.0
    ) -> ServiceConfig:
        durations = gpu_step_durations()
        mean_item_s = (sum(durations) / len(durations)) if durations else 10.0
        return cls(
            max_gpu_backlog=int(
                settings.max_workers * max_acceptable_wait_s / mean_item_s
            ),
            max_acceptable_wait_s=max_acceptable_wait_s,
        )


class JobService:
    """Facade: submit / status / cancel / metrics / system status."""

    def __init__(
        self,
        *,
        settings: KitSettings | None = None,
        clock: Clock | None = None,
        pool: GpuPool | None = None,
        store: JobStore | None = None,
        policy: WarmPoolPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
        config: ServiceConfig | None = None,
    ) -> None:
        configure_logging()
        self.settings = settings or KitSettings()
        self.clock = clock or RealClock(self.settings.time_scale)
        self.pool = pool or GpuPool.from_settings(self.settings, self.clock)
        self.store: JobStore = store or InMemoryJobStore()
        self.config = config or ServiceConfig.from_settings(self.settings)
        self.metrics = MetricsRegistry()
        retry = retry_policy or RetryPolicy()
        gpu_durations = gpu_step_durations()
        self.scheduler = GpuScheduler(
            pool=self.pool,
            clock=self.clock,
            policy=policy or AdaptivePolicy(),
            retry_policy=retry,
            metrics=self.metrics,
            max_backlog=self.config.max_gpu_backlog,
            nominal_service_s=(
                sum(gpu_durations) / len(gpu_durations) if gpu_durations else 10.0
            ),
            expected_cold_start_s=(
                (self.settings.cold_start_min_s + self.settings.cold_start_max_s) / 2
            ),
            rng_seed=self.settings.seed,
        )
        self.runner = ChainRunner(
            settings=self.settings,
            clock=self.clock,
            scheduler=self.scheduler,
            retry_policy=retry,
            metrics=self.metrics,
            ledger=AttemptLedger(),
        )
        self._job_tasks: dict[str, asyncio.Task[None]] = {}

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.scheduler.start()

    async def stop(self) -> None:
        """Stop runners and scheduler; in-flight jobs resolve as cancelled."""
        await self.scheduler.stop()
        # Let the JobCancelledErrors just set on queued-item futures deliver
        # to their awaiting job tasks before we cancel those tasks, so the
        # exceptions are consumed rather than logged as never-retrieved.
        await asyncio.sleep(0)
        for task in list(self._job_tasks.values()):
            task.cancel()
        await asyncio.gather(*self._job_tasks.values(), return_exceptions=True)
        self._job_tasks.clear()

    # ----------------------------------------------------------------- submit

    def submit(
        self,
        chain: list[str] | str,
        image: Image,
        idempotency_key: str | None = None,
    ) -> tuple[Job, bool]:
        """Validate, dedup, admit, and start a job. Returns (job, dedup_hit).

        Raises ChainValidationError (bad chain) or OverCapacityError (GPU
        backlog exceeds the admission bound).
        """
        if not self.scheduler.started:
            raise ServiceError(
                "service not started: call start() (or run via the API "
                "lifespan) before submitting jobs"
            )
        steps = resolve_chain(chain)
        validate_chain(steps)

        key = idempotency_key or derived_idempotency_key(steps, image)
        existing = self.store.get_by_idempotency_key(key)
        # FAILED and CANCELLED jobs may be resubmitted — the agent contract
        # explicitly suggests resubmit for both, so dedup must not trap the
        # caller on a dead job.
        resubmittable = {JobState.FAILED, JobState.CANCELLED}
        if existing is not None and existing.state not in resubmittable:
            self.metrics.inc("submits_deduped")
            return existing, True

        job_id = str(uuid.uuid4())
        if not self.scheduler.try_admit(job_id, gpu_item_count(steps)):
            self.metrics.inc("submits_rejected")
            retry_after = self.scheduler.estimate_wait_s()
            log_event(
                "queue.rejected",
                admitted_backlog=self.scheduler.admitted_backlog,
                retry_after_s=round(retry_after, 1),
            )
            raise OverCapacityError(retry_after, self.scheduler.admitted_backlog)

        job = Job(
            job_id=job_id,
            idempotency_key=key,
            chain=steps,
            image=image,
            created_at=self.clock.monotonic(),
            steps=[StepRun(step_name=name) for name in steps],
        )
        self.store.put(job)
        self.metrics.inc("submits_accepted")
        log_event(
            "job.submitted", job_id=job.job_id, chain=list(steps), image=image.id
        )
        task = asyncio.create_task(self.runner.run(job))
        self._job_tasks[job.job_id] = task
        task.add_done_callback(functools.partial(self._on_done, job.job_id))
        return job, False

    def _on_done(self, job_id: str, _task: asyncio.Task[None]) -> None:
        self._job_tasks.pop(job_id, None)
        # Release any admitted GPU capacity the job never turned into items
        # (failed early, cancelled, or its GPU steps were memoized).
        self.scheduler.finalize_job(job_id)
        job = self.store.get(job_id)
        if job is not None and job.is_terminal:
            self.metrics.observe_job(job)

    # ----------------------------------------------------------------- queries

    def get_job(self, job_id: str) -> Job | None:
        return self.store.get(job_id)

    def list_jobs(self) -> list[Job]:
        return self.store.list_jobs()

    def all_terminal(self, strict: bool = False) -> bool:
        """Return True when no job is still running.

        O(1) via the live-task registry (ChainRunner guarantees a task never
        finishes with a non-terminal job); ``strict=True`` cross-checks the
        store — used by tests to prove that guarantee holds.
        """
        if strict:
            return all(job.is_terminal for job in self.store.list_jobs())
        return not self._job_tasks

    def trace(self, job_id: str) -> dict[str, Any] | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        return build_trace(job, self.settings.cost_per_second)

    def estimated_wait_s(self) -> float:
        return self.scheduler.estimate_wait_s()

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot(
            self.pool,
            queue_depth=self.scheduler.queue_depth,
            admitted_backlog=self.scheduler.admitted_backlog,
            warm_target=self.scheduler.warm_target_now,
            avg_gpu_service_s=self.scheduler.avg_service_s,
            estimated_wait_s=self.estimated_wait_s(),
        )

    def system_status(self) -> dict[str, Any]:
        """Report the pre-submit signal (should a caller submit right now?)."""
        backlog = self.scheduler.admitted_backlog
        return {
            "queue_depth": self.scheduler.queue_depth,
            "admitted_backlog": backlog,
            "backlog_capacity": self.scheduler.max_backlog,
            "accepting_gpu_jobs": backlog < self.scheduler.max_backlog,
            "estimated_wait_s": self.estimated_wait_s(),
            "gpu_capacity": self.pool.capacity,
            **gpu_economics(self.pool.snapshot()),
        }

    # ----------------------------------------------------------------- cancel

    def cancel(self, job_id: str) -> Job | None:
        """Cancel a job, best-effort.

        Queued work is removed; an in-flight GPU attempt runs out (the kit
        offers no preemption) and the job stops before its next step.
        """
        job = self.store.get(job_id)
        if job is None or job.is_terminal:
            return job
        job.cancel_requested = True
        item = self.runner.active_item(job_id)
        if item is not None and not item.future.done():
            # Always flag the item: an in-flight step finishes its current
            # attempt but must not burn the rest of its retry budget.
            item.cancelled = True
            if not item.started:
                # Still queued: pull it out now so the runner marks the job
                # cancelled immediately instead of waiting its turn (its
                # admission reservation is released by the future callback).
                item.future.set_exception(
                    JobCancelledError("cancelled while queued")
                )
        log_event("job.cancel_requested", job_id=job_id)
        return job
