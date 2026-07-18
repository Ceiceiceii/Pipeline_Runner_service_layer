"""Shared fixtures for the kit's substrate-trust tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import pytest

from pipeline_kit.clock import ManualClock

T = TypeVar("T")

# A coroutine runner that advances a ManualClock until the coroutine finishes.
RunWithClock = Callable[[ManualClock, Awaitable[T]], Awaitable[T]]


@pytest.fixture
def manual_clock() -> ManualClock:
    """A fresh virtual clock starting at t=0."""
    return ManualClock()


@pytest.fixture
def run_with_clock() -> RunWithClock[object]:
    """Return a helper that runs a coroutine, advancing the clock as it sleeps."""

    async def _run(
        clock: ManualClock,
        coro: Awaitable[T],
        *,
        step: float = 1.0,
        max_advance: float = 1_000_000.0,
    ) -> T:
        task: asyncio.Task[T] = asyncio.ensure_future(coro)
        await asyncio.sleep(0)  # let the coroutine reach its first sleep
        advanced = 0.0
        while not task.done() and advanced < max_advance:
            await clock.advance(step)
            await asyncio.sleep(0)
            advanced += step
        return await task

    return _run
