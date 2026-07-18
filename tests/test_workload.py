"""Tests for the burst workload generator: reproducibility, spikiness, replay."""

from __future__ import annotations

import asyncio

from pipeline_kit.config import KitSettings
from pipeline_kit.workload import BurstWorkload, Request


def test_schedule_is_reproducible():
    settings = KitSettings(seed=42)
    first = BurstWorkload(settings).schedule()
    second = BurstWorkload(settings).schedule()
    assert [r.t_offset for r in first] == [r.t_offset for r in second]
    assert [r.chain for r in first] == [r.chain for r in second]


def test_schedule_is_spiky():
    settings = KitSettings(
        seed=1,
        base_rate=0.2,
        burst_rate=5.0,
        quiet_duration_s=30.0,
        burst_duration_s=10.0,
        n_bursts=2,
    )
    schedule = BurstWorkload(settings).schedule()
    quiet = sum(1 for r in schedule if 0.0 <= r.t_offset < 30.0)
    burst = sum(1 for r in schedule if 30.0 <= r.t_offset < 40.0)
    # A 10s burst at 5/s far out-densities a 30s quiet window at 0.2/s.
    assert burst > quiet


def test_schedule_offsets_are_sorted_and_indexed():
    schedule = BurstWorkload(KitSettings(seed=5)).schedule()
    offsets = [r.t_offset for r in schedule]
    assert offsets == sorted(offsets)
    assert [r.index for r in schedule] == list(range(len(schedule)))


async def test_drive_invokes_every_request(manual_clock):
    settings = KitSettings(
        seed=3,
        base_rate=1.0,
        burst_rate=3.0,
        quiet_duration_s=5.0,
        burst_duration_s=5.0,
        n_bursts=1,
    )
    workload = BurstWorkload(settings)
    schedule = workload.schedule()
    assert schedule  # the chosen knobs produce arrivals

    seen: list[int] = []

    async def submit(request: Request) -> int:
        seen.append(request.index)
        return request.index

    drive_task = asyncio.ensure_future(workload.drive(submit, clock=manual_clock))
    await asyncio.sleep(0)
    for _ in range(1000):
        if drive_task.done():
            break
        await manual_clock.advance(1.0)
        await asyncio.sleep(0)
    results = await drive_task

    assert sorted(seen) == [r.index for r in schedule]
    assert sorted(results) == [r.index for r in schedule]
