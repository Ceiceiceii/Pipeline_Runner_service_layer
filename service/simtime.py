"""One ManualClock advance loop for every consumer (harness and tests).

``ManualClock.advance`` only yields to the event loop when it releases due
sleepers, so tasks created between sleeps (e.g. freshly spawned worker-runner
tasks) can starve unless the driver yields explicitly. That subtlety must
live in exactly one place: a fix applied to a private copy of this loop and
not the others would make the harness and the tests settle under different
scheduling semantics.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from pipeline_kit.clock import ManualClock


async def advance_until(
    clock: ManualClock,
    done: Callable[[], bool],
    *,
    step_s: float = 1.0,
    max_s: float = 36_000.0,
    label: str = "condition",
) -> float:
    """Advance virtual time until ``done()`` is true; return seconds advanced.

    Raises RuntimeError if ``max_s`` of virtual time passes first.
    """
    advanced = 0.0
    await asyncio.sleep(0)  # let just-created tasks reach their first await
    while not done():
        if advanced >= max_s:
            raise RuntimeError(f"{label}: not reached after {max_s}s of virtual time")
        await clock.advance(step_s)
        # advance() only yields when it releases sleepers; yield twice so
        # tasks created during this tick (and tasks they create) get to run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        advanced += step_s
    # The condition may flip in the same pass that scheduled completion
    # callbacks (task done-callbacks fire via call_soon); drain them so the
    # caller observes their effects too.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return advanced
