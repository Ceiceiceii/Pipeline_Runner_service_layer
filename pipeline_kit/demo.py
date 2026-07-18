"""Illustrative smoke run of the kit's primitives — **NOT a scheduler**.

This runs the four steps once and prints per-step timing plus the GPU cost
incurred. It does the simplest possible thing: warm one worker, run the two GPU
steps on it, cool it. There is deliberately no scheduling logic here — no
warm-pool policy, no retries, no queue, no metrics. Building that is the
exercise; this file only proves the substrate works and shows how the pieces fit
together.

Run it with ``python -m pipeline_kit.demo``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from pipeline_kit.clock import Clock, RealClock
from pipeline_kit.config import KitSettings
from pipeline_kit.gpu import GpuPool
from pipeline_kit.pipelines import fit_to_last, generate_multiview, remove_bg, segment
from pipeline_kit.schemas import Image

T = TypeVar("T")


async def _timed(clock: Clock, coro: Awaitable[T]) -> tuple[T, float]:
    """Await ``coro`` and return ``(result, simulated_seconds_elapsed)``."""
    start = clock.monotonic()
    result = await coro
    return result, clock.monotonic() - start


async def main() -> None:
    """Run one full chain and print timings + cost."""
    # Force reliable, time-compressed settings so the illustration always
    # completes cleanly and quickly. Flakiness and retries are your concern.
    settings = KitSettings(
        segment_failure_rate=0.0,
        remove_bg_failure_rate=0.0,
        multiview_failure_rate=0.0,
        fit_to_last_failure_rate=0.0,
        time_scale=30.0,
    )
    clock = RealClock(settings.time_scale)
    image = Image(id="demo-image")

    print("=== Pipeline Runner kit - illustrative run (NOT a scheduler) ===")
    print(f"(simulated time compressed {settings.time_scale:.0f}x; seconds shown are simulated)\n")

    mask, t_seg = await _timed(clock, segment(image, settings=settings, clock=clock))
    print(f"  segment      (CPU)  {t_seg:6.2f}s  -> mask   {mask.id}")
    cutout, t_bg = await _timed(
        clock, remove_bg(image, mask, settings=settings, clock=clock)
    )
    print(f"  remove_bg    (CPU)  {t_bg:6.2f}s  -> cutout {cutout.id}")

    pool = GpuPool.from_settings(settings, clock)
    with pool.lease() as worker:
        _, t_warm = await _timed(clock, worker.warm())
        print(f"  warm worker {worker.id}       {t_warm:6.2f}s  (cold-start penalty, billed)")
        views, t_mv = await _timed(
            clock, worker.run(generate_multiview(cutout, settings=settings, clock=clock))
        )
        print(f"  multiview    (GPU)  {t_mv:6.2f}s  -> {len(views.views)} views")
        mesh, t_fit = await _timed(
            clock, worker.run(fit_to_last(views, settings=settings, clock=clock))
        )
        print(f"  fit_to_last  (GPU)  {t_fit:6.2f}s  -> mesh   {mesh.id} ({mesh.vertex_count} verts)")
        worker.cool()

    report = pool.cost_report()
    print(
        f"\nGPU cost: ${report.total_cost:.4f}  "
        f"({report.warm_worker_seconds:.1f} warm worker-seconds, "
        f"{report.utilization * 100:.0f}% utilization)"
    )
    print("CPU steps cost nothing and use no pool. Cold start dominated the bill -")
    print("amortizing it across work is exactly what a warm-pool policy is for.")


if __name__ == "__main__":
    asyncio.run(main())
