"""End-to-end: the full default burst schedule against the whole service."""

from __future__ import annotations

import asyncio

from pipeline_kit.clock import ManualClock
from pipeline_kit.config import KitSettings
from pipeline_kit.workload import BurstWorkload, Request
from service.models import OverCapacityError
from service.service import JobService
from tests.service.conftest import settle


async def test_full_burst_replay_default_settings():
    settings = KitSettings(seed=0)  # jitter on: the real deal
    clock = ManualClock()
    service = JobService(settings=settings, clock=clock)
    service.start()

    rejected = 0
    retry_after_hints: list[float] = []

    async def submit(request: Request) -> str | None:
        nonlocal rejected
        try:
            job, _ = service.submit(list(request.chain), request.image)
        except OverCapacityError as exc:
            rejected += 1
            retry_after_hints.append(exc.retry_after_s)
            return None
        return job.job_id

    workload = BurstWorkload(settings)
    total_requests = len(workload.schedule())
    drive = asyncio.create_task(workload.drive(submit, clock=clock))
    await settle(
        clock,
        until=lambda: drive.done() and service.all_terminal(),
        max_advance=20_000,
    )
    await drive

    jobs = service.list_jobs()
    # Dedup can collapse repeat images, but every admitted job reached a
    # terminal state and admissions + rejections cover the whole schedule.
    accepted = service.metrics.counters["submits_accepted"]
    deduped = service.metrics.counters["submits_deduped"]
    assert accepted + deduped + rejected == total_requests
    assert all(job.is_terminal for job in jobs)

    snapshot = service.metrics_snapshot()
    succeeded = snapshot["counters"]["jobs_succeeded"]
    assert succeeded > 50  # the bulk of admitted work completed

    # The demand is ~5x oversubscribed during the arrival window, so the
    # bounded queue must have engaged: rejection is designed behavior here.
    assert rejected > 0
    assert all(hint > 0 for hint in retry_after_hints)

    # Economics sanity: money was spent, but less than holding the whole
    # pool warm for the entire run (the always-warm cost ceiling).
    makespan = clock.monotonic()
    total_cost = snapshot["gpu"]["total_cost"]
    always_warm_ceiling = (
        settings.max_workers * makespan * settings.cost_per_second
    )
    assert 0 < total_cost < always_warm_ceiling
    assert snapshot["gpu"]["utilization"] > 0.5  # warm workers mostly worked

    # Percentiles exist and reflect real queueing (minutes, not seconds).
    assert snapshot["job_latency_s"]["p50"] is not None
    assert snapshot["job_latency_s"]["p95"] >= snapshot["job_latency_s"]["p50"]

    await service.stop()
    # After stop, every worker is cooled: the bill has stopped growing.
    final_cost = service.pool.snapshot().cost.total_cost
    await clock.advance(100)
    assert service.pool.snapshot().cost.total_cost == final_cost
