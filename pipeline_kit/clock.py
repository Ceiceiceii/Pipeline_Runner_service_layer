"""Clock abstraction shared by the GPU pool and the workload driver.

GPU cost is a function of *time spent warm*, and the workload generator replays
arrivals over *time*. If those two read different clocks — say, a time-compressed
replay against a wall-clock cost meter — every dollar figure comes out wrong. So
there is exactly one ``Clock`` in the system, injected everywhere time is read or
awaited.

Two implementations ship with the kit:

* :class:`RealClock` — wall-clock time, optionally compressed by ``time_scale``
  so a 60-second cold start does not make a demo take a minute.
* :class:`ManualClock` — virtual time you advance explicitly, so tests and
  headless simulations run instantly *and* deterministically.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """A monotonic time source with an awaitable sleep."""

    def monotonic(self) -> float:
        """Return monotonically increasing time, in (possibly scaled) seconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for ``seconds`` of this clock's time."""
        ...


class RealClock:
    """Wall-clock implementation backed by :func:`asyncio.sleep`.

    ``time_scale`` compresses simulated time: at ``time_scale=60`` a 60-second
    cold start really sleeps for one second, and :meth:`monotonic` advances 60
    simulated seconds per real second. Because cost is computed from this same
    clock, the reported dollar cost reflects the full simulated duration.
    """

    def __init__(self, time_scale: float = 1.0) -> None:
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self._scale = time_scale
        self._origin = time.monotonic()

    def monotonic(self) -> float:
        """Return scaled simulated seconds since this clock was created."""
        return (time.monotonic() - self._origin) * self._scale

    async def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds`` of simulated time (``seconds / time_scale`` real)."""
        await asyncio.sleep(max(0.0, seconds) / self._scale)


class ManualClock:
    """Virtual clock advanced explicitly via :meth:`advance`.

    Sleeps register a deadline and block until virtual time reaches it. Nothing
    happens in real time, so simulations and tests are both instant and exact.
    Use the async :meth:`advance` so woken coroutines get a chance to run.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def monotonic(self) -> float:
        """Return the current virtual time."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Block until virtual time has advanced by ``seconds``."""
        if seconds <= 0:
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._sleepers.append((self._now + seconds, future))
        await future

    async def advance(self, seconds: float) -> None:
        """Advance virtual time by ``seconds``, releasing any due sleepers.

        Sleepers are released in deadline order; after each batch we yield to the
        event loop so woken coroutines can run (and register new sleeps) before
        time advances further.
        """
        target = self._now + seconds
        while True:
            due = [(d, f) for d, f in self._sleepers if d <= target and not f.done()]
            if not due:
                break
            next_deadline = min(d for d, _ in due)
            self._now = next_deadline
            for deadline, future in due:
                if deadline <= next_deadline and not future.done():
                    future.set_result(None)
            self._sleepers = [(d, f) for d, f in self._sleepers if not f.done()]
            await asyncio.sleep(0)  # let woken coroutines make progress
        self._now = target
