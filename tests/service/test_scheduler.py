"""Scheduler economics: coalescing, scale-up, TTL cool-down, retry-in-place."""

from __future__ import annotations

from pipeline_kit.schemas import Image
from service.models import JobState
from service.scheduler import AdaptivePolicy, AlwaysWarmN
from tests.service.conftest import make_service, settle


def _gpu_ok_overrides() -> dict:
    return {"multiview_failure_rate": 0.0, "fit_to_last_failure_rate": 0.0}


async def test_coalescing_many_jobs_one_cold_start(clock):
    """A single warm worker drains multiple jobs across one warm interval."""
    service = make_service(
        clock,
        policy=AlwaysWarmN(1),
        settings_overrides=_gpu_ok_overrides(),
    )
    service.start()
    jobs = [service.submit("multiview", Image(id=f"img-{i}"))[0] for i in range(3)]
    await settle(clock, until=lambda: all(j.is_terminal for j in jobs))
    assert all(j.state is JobState.SUCCEEDED for j in jobs)
    assert service.metrics.counters["cold_starts"] == 1
    # All GPU attempts ran on the same worker slot.
    worker_ids = {
        span.worker_id
        for job in jobs
        for step in job.steps
        for span in step.attempts
        if span.worker_id is not None
    }
    assert len(worker_ids) == 1
    await service.stop()


async def test_scale_up_reaches_full_pool_under_backlog(clock):
    service = make_service(
        clock,
        policy=AdaptivePolicy(target_drain_s=60.0, idle_ttl_s=60.0),
        settings_overrides=_gpu_ok_overrides(),
    )
    service.start()
    # 24 jobs => 24 queued GPU items x ~10s each: a >60s projected drain even
    # on the full pool, so the policy must drive straight to all 4 workers.
    jobs = [service.submit("full", Image(id=f"img-{i}"))[0] for i in range(24)]
    await settle(clock, until=lambda: service.pool.warm_count == 4, step=0.5)
    await settle(clock, until=lambda: all(j.is_terminal for j in jobs))
    assert all(j.state is JobState.SUCCEEDED for j in jobs)
    assert service.metrics.counters["cold_starts"] == 4
    await service.stop()


async def test_idle_ttl_cools_workers_and_cost_stops(clock):
    service = make_service(
        clock,
        policy=AdaptivePolicy(target_drain_s=60.0, idle_ttl_s=30.0),
        settings_overrides=_gpu_ok_overrides(),
    )
    service.start()
    job, _ = service.submit("multiview", Image(id="img-1"))
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.SUCCEEDED
    assert service.pool.warm_count == 1
    # TTL passes with an empty queue: the worker cools and billing stops.
    await settle(clock, until=lambda: service.pool.warm_count == 0, max_advance=40)
    cooled_cost = service.pool.snapshot().cost.total_cost
    await clock.advance(500)
    assert service.pool.snapshot().cost.total_cost == cooled_cost
    await service.stop()


async def test_min_warm_floor_survives_ttl(clock):
    service = make_service(
        clock,
        policy=AdaptivePolicy(idle_ttl_s=20.0, min_warm=1),
        settings_overrides=_gpu_ok_overrides(),
    )
    service.start()
    await settle(clock, until=lambda: service.pool.warm_count == 1, max_advance=100)
    job, _ = service.submit("multiview", Image(id="img-1"))
    await settle(clock, until=lambda: job.is_terminal)
    await clock.advance(200)  # many TTLs later the floor worker is still warm
    assert service.pool.warm_count == 1
    await service.stop()


async def test_retries_happen_in_place_on_the_held_worker(clock):
    service = make_service(
        clock,
        policy=AlwaysWarmN(1),
        settings_overrides={"multiview_failure_rate": 1.0},
    )
    service.start()
    job, _ = service.submit("multiview", Image(id="img-1"))
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.FAILED
    gpu_step = next(s for s in job.steps if s.step_name == "generate_multiview")
    assert len(gpu_step.attempts) == 4
    # Every retry ran on the same worker: no requeue, no second cold start.
    assert len({span.worker_id for span in gpu_step.attempts}) == 1
    assert service.metrics.counters["cold_starts"] == 1
    await service.stop()


async def test_failed_attempts_still_bill(clock):
    service = make_service(
        clock,
        policy=AlwaysWarmN(1),
        settings_overrides={"multiview_failure_rate": 1.0},
    )
    service.start()
    job, _ = service.submit("multiview", Image(id="img-1"))
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.FAILED
    assert service.pool.snapshot().cost.total_cost > 0
    await service.stop()
