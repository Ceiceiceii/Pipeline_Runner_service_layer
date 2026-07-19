"""Shared fixtures for the service-layer tests (all on ManualClock)."""

from __future__ import annotations

import logging

import pytest

from pipeline_kit.clock import ManualClock
from pipeline_kit.config import KitSettings
from service.service import JobService
from service.simtime import advance_until


@pytest.fixture(autouse=True)
def _quiet_logs() -> None:
    logging.getLogger("service").setLevel(logging.WARNING)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


def make_service(clock: ManualClock, **overrides) -> JobService:
    """A service on virtual time. Jitter off by default so timings are exact."""
    setting_overrides = overrides.pop("settings_overrides", {})
    settings = KitSettings(latency_jitter=0.0, **setting_overrides)
    return JobService(settings=settings, clock=clock, **overrides)


async def settle(
    clock: ManualClock,
    *,
    until,
    step: float = 1.0,
    max_advance: float = 100_000.0,
) -> float:
    """Advance virtual time until ``until()`` is true; return seconds advanced.

    Thin wrapper over the one shared advance loop (service.simtime), so tests
    and the harness settle under identical scheduling semantics.
    """
    return await advance_until(
        clock, until, step_s=step, max_s=max_advance, label="test condition"
    )
