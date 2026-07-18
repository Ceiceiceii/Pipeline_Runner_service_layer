"""Substrate-trust tests for the GPU pool: cost, cold start, and boundaries.

If these numbers are wrong, every candidate's scheduling and cost work is built
on sand — so they assert the economic edges exactly, using a virtual clock.
"""

from __future__ import annotations

import asyncio

import pytest

from pipeline_kit.clock import ManualClock
from pipeline_kit.errors import (
    PoolExhaustedError,
    WorkerBusyError,
    WorkerNotReadyError,
)
from pipeline_kit.gpu import GpuPool, Worker, WorkerState


def _worker(
    clock: ManualClock,
    *,
    cold_start: float = 10.0,
    cost: float = 1.0,
    seed: int = 0,
) -> Worker:
    return Worker(
        0,
        cold_start_min_s=cold_start,
        cold_start_max_s=cold_start,
        cost_per_second=cost,
        clock=clock,
        seed=seed,
    )


async def test_warm_pays_cold_start_before_idle(manual_clock):
    worker = _worker(manual_clock, cold_start=30.0)
    task = asyncio.ensure_future(worker.warm())
    await asyncio.sleep(0)
    assert worker.state == WorkerState.WARMING
    await manual_clock.advance(29.0)
    assert worker.state == WorkerState.WARMING  # cold start not finished
    await manual_clock.advance(1.0)
    await task
    assert worker.state == WorkerState.IDLE


async def test_warm_then_cool_with_no_job_still_bills_cold_start(
    manual_clock, run_with_clock
):
    worker = _worker(manual_clock, cold_start=40.0, cost=1.0)
    await run_with_clock(manual_clock, worker.warm())
    worker.cool()  # never ran a single job
    assert worker.cost() == pytest.approx(40.0)
    assert worker.busy_seconds() == pytest.approx(0.0)


async def test_cost_is_warm_interval_not_a_sum_over_jobs(manual_clock, run_with_clock):
    worker = _worker(manual_clock, cold_start=10.0, cost=1.0)
    await run_with_clock(manual_clock, worker.warm())

    async def job() -> str:
        await manual_clock.sleep(5.0)
        return "ok"

    assert await run_with_clock(manual_clock, worker.run(job())) == "ok"
    assert await run_with_clock(manual_clock, worker.run(job())) == "ok"
    worker.cool()
    # Warm wall-time: 10 (cold) + 5 + 5 = 20. Cost tracks warm time, not jobs.
    assert worker.warm_seconds() == pytest.approx(20.0)
    assert worker.busy_seconds() == pytest.approx(10.0)
    assert worker.cost() == pytest.approx(20.0)


async def test_run_on_cold_worker_raises(manual_clock):
    worker = _worker(manual_clock)

    async def job() -> int:
        return 1

    coro = job()
    with pytest.raises(WorkerNotReadyError):
        await worker.run(coro)
    coro.close()  # run() raised before awaiting it


async def test_run_on_busy_worker_raises(manual_clock, run_with_clock):
    worker = _worker(manual_clock, cold_start=1.0)
    await run_with_clock(manual_clock, worker.warm())

    async def slow() -> str:
        await manual_clock.sleep(100.0)
        return "done"

    task = asyncio.ensure_future(worker.run(slow()))
    await asyncio.sleep(0)
    assert worker.state == WorkerState.BUSY

    async def other() -> int:
        return 2

    coro = other()
    with pytest.raises(WorkerBusyError):
        await worker.run(coro)
    coro.close()

    await manual_clock.advance(100.0)
    assert await task == "done"
    assert worker.state == WorkerState.IDLE


async def test_failing_job_bills_warm_time_and_releases_worker(
    manual_clock, run_with_clock
):
    worker = _worker(manual_clock, cold_start=10.0, cost=1.0)
    await run_with_clock(manual_clock, worker.warm())

    async def boom() -> None:
        await manual_clock.sleep(5.0)
        raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError):
        await run_with_clock(manual_clock, worker.run(boom()))

    assert worker.state == WorkerState.IDLE  # released despite the failure
    worker.cool()
    assert worker.busy_seconds() == pytest.approx(5.0)  # failed job still occupied it
    assert worker.cost() == pytest.approx(15.0)  # warm interval [0, 15]


def test_acquire_raises_when_pool_exhausted(manual_clock):
    pool = GpuPool(
        max_workers=2,
        cold_start_min_s=1.0,
        cold_start_max_s=1.0,
        cost_per_second=1.0,
        clock=manual_clock,
    )
    first = pool.acquire()
    pool.acquire()
    with pytest.raises(PoolExhaustedError):
        pool.acquire()
    pool.release(first)
    assert pool.acquire() is first


async def test_pool_cost_report_and_utilization(manual_clock, run_with_clock):
    pool = GpuPool(
        max_workers=2,
        cold_start_min_s=10.0,
        cold_start_max_s=10.0,
        cost_per_second=2.0,
        clock=manual_clock,
    )
    worker = pool.acquire()
    await run_with_clock(manual_clock, worker.warm())

    async def job() -> None:
        await manual_clock.sleep(10.0)

    await run_with_clock(manual_clock, worker.run(job()))

    report = pool.cost_report()
    assert report.warm_worker_seconds == pytest.approx(20.0)
    assert report.busy_worker_seconds == pytest.approx(10.0)
    assert report.total_cost == pytest.approx(40.0)
    assert report.utilization == pytest.approx(0.5)
    assert 0.0 <= report.utilization <= 1.0
    assert report.burn_rate_per_second == pytest.approx(2.0)  # one warm worker
