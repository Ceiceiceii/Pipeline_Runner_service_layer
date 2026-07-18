"""Bursty workload generator.

Real demand arrives in spikes, not at a steady rate, and a scheduler's cost and
latency only look interesting under that shape. This generator produces a
**spiky** arrival schedule — a quiet Poisson baseline punctuated by high-rate
bursts — and replays it against your service.

The schedule is *materialized up-front* from the seed (synchronously, before any
concurrency), so a given seed always yields the identical arrival pattern no
matter how the async replay interleaves.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pipeline_kit.clock import Clock, RealClock
from pipeline_kit.config import (
    FIT_TO_LAST,
    GENERATE_MULTIVIEW,
    REMOVE_BG,
    SEGMENT,
    KitSettings,
)
from pipeline_kit.determinism import rng_for
from pipeline_kit.schemas import Image

# Chains the two consumers tend to ask for, with rough relative frequencies.
FULL_CHAIN = (SEGMENT, REMOVE_BG, GENERATE_MULTIVIEW, FIT_TO_LAST)
MULTIVIEW_CHAIN = (SEGMENT, REMOVE_BG, GENERATE_MULTIVIEW)
CUTOUT_CHAIN = (SEGMENT, REMOVE_BG)
_CHAIN_WEIGHTS: tuple[tuple[tuple[str, ...], float], ...] = (
    (FULL_CHAIN, 0.6),
    (MULTIVIEW_CHAIN, 0.25),
    (CUTOUT_CHAIN, 0.15),
)


@dataclass(frozen=True)
class Request:
    """One arrival: run ``chain`` on ``image`` at ``t_offset`` seconds from start."""

    index: int
    t_offset: float
    chain: tuple[str, ...]
    image: Image


async def _invoke(
    submit: Callable[[Request], Awaitable[object]],
    request: Request,
) -> object:
    return await submit(request)


class BurstWorkload:
    """Generates and replays a spiky arrival schedule."""

    def __init__(self, settings: KitSettings | None = None) -> None:
        self._settings = settings or KitSettings()

    def schedule(self) -> list[Request]:
        """Materialize the full timed arrival list deterministically."""
        settings = self._settings
        rng = rng_for(settings.seed, "workload")
        requests: list[Request] = []
        cursor = 0.0
        for _cycle in range(settings.n_bursts):
            cursor = self._emit(
                requests, rng, start=cursor,
                duration=settings.quiet_duration_s, rate=settings.base_rate,
            )
            cursor = self._emit(
                requests, rng, start=cursor,
                duration=settings.burst_duration_s, rate=settings.burst_rate,
            )
        requests.sort(key=lambda request: request.t_offset)
        return [
            Request(index=i, t_offset=r.t_offset, chain=r.chain, image=r.image)
            for i, r in enumerate(requests)
        ]

    def _emit(
        self,
        requests: list[Request],
        rng: random.Random,
        *,
        start: float,
        duration: float,
        rate: float,
    ) -> float:
        """Append Poisson arrivals over ``[start, start + duration)``; return the end."""
        end = start + duration
        if rate <= 0:
            return end
        cursor = start
        while True:
            cursor += rng.expovariate(rate)
            if cursor >= end:
                return end
            index = len(requests)
            requests.append(
                Request(
                    index=index,
                    t_offset=cursor,
                    chain=self._pick_chain(rng),
                    image=Image(id=f"img-{index:05d}"),
                ),
            )

    @staticmethod
    def _pick_chain(rng: random.Random) -> tuple[str, ...]:
        roll = rng.random()
        cumulative = 0.0
        for chain, weight in _CHAIN_WEIGHTS:
            cumulative += weight
            if roll < cumulative:
                return chain
        return FULL_CHAIN

    async def drive(
        self,
        submit: Callable[[Request], Awaitable[object]],
        *,
        clock: Clock | None = None,
    ) -> list[object]:
        """Replay the schedule, calling ``submit(request)`` at each arrival time.

        ``submit`` is your service's entry point; its return values (e.g. job
        ids) are collected and returned. Submissions are fire-and-forget, so a
        slow ``submit`` never delays a later arrival.
        """
        clock = clock or RealClock(self._settings.time_scale)
        start = clock.monotonic()
        pending: list[asyncio.Task[object]] = []
        for request in self.schedule():
            wait = request.t_offset - (clock.monotonic() - start)
            if wait > 0:
                await clock.sleep(wait)
            pending.append(asyncio.create_task(_invoke(submit, request)))
        return list(await asyncio.gather(*pending))
