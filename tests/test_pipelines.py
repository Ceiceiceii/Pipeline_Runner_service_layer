"""Tests for the pipeline functions: determinism, content ids, types, jitter."""

from __future__ import annotations

from itertools import pairwise

import pytest
from pydantic import ValidationError

from pipeline_kit import STEPS
from pipeline_kit.clock import ManualClock
from pipeline_kit.config import (
    FIT_TO_LAST,
    GENERATE_MULTIVIEW,
    REMOVE_BG,
    SEGMENT,
    KitSettings,
)
from pipeline_kit.errors import TransientError
from pipeline_kit.pipelines import (
    _jittered_duration,
    generate_multiview,
    remove_bg,
    segment,
)
from pipeline_kit.schemas import Cutout, Image, MultiviewResult, View


async def test_transient_failure_is_deterministic():
    settings = KitSettings(
        multiview_duration_s=0.0,
        latency_jitter=0.0,
        multiview_failure_rate=0.5,
        seed=7,
    )
    clock = ManualClock()
    cutouts = [Cutout(id=f"c{i}", image_id="i", mask_id="m") for i in range(24)]

    async def outcomes() -> list[bool]:
        results: list[bool] = []
        for cutout in cutouts:
            try:
                await generate_multiview(cutout, settings=settings, clock=clock)
                results.append(True)
            except TransientError:
                results.append(False)
        return results

    first = await outcomes()
    second = await outcomes()
    assert first == second  # identical regardless of interleaving / run
    assert any(first)
    assert not all(first)  # a 0.5 rate yields a mix


async def test_incrementing_attempt_can_clear_a_transient_failure():
    settings = KitSettings(
        multiview_duration_s=0.0,
        latency_jitter=0.0,
        multiview_failure_rate=0.5,
        seed=3,
    )
    clock = ManualClock()
    cutout = Cutout(id="retry-me", image_id="i", mask_id="m")

    succeeded_at: int | None = None
    for attempt in range(40):
        try:
            await generate_multiview(
                cutout, settings=settings, clock=clock, attempt=attempt
            )
            succeeded_at = attempt
            break
        except TransientError:
            continue
    assert succeeded_at is not None  # retries have an independent chance to clear


async def test_output_ids_are_content_addressed():
    settings = KitSettings(segment_duration_s=0.0, latency_jitter=0.0)
    clock = ManualClock()
    image = Image(id="abc")

    mask_a = await segment(image, settings=settings, clock=clock)
    mask_b = await segment(image, settings=settings, clock=clock)
    assert mask_a.id == mask_b.id  # same input -> same id (idempotency hook)
    assert mask_a.image_id == "abc"

    other = await segment(Image(id="xyz"), settings=settings, clock=clock)
    assert other.id != mask_a.id


async def test_cpu_steps_are_always_available():
    settings = KitSettings(
        segment_duration_s=0.0,
        remove_bg_duration_s=0.0,
        latency_jitter=0.0,
        seed=99,
    )
    clock = ManualClock()
    for i in range(50):
        image = Image(id=f"img{i}")
        mask = await segment(image, settings=settings, clock=clock)
        cutout = await remove_bg(image, mask, settings=settings, clock=clock)
        assert cutout.image_id == image.id  # never raises at the default 0.0 rate


def test_multiview_requires_exactly_eight_views():
    too_few = [View(id=f"v{i}", index=i, azimuth=0.0) for i in range(3)]
    with pytest.raises(ValidationError):
        MultiviewResult(id="x", cutout_id="c", views=too_few)

    exactly_eight = [View(id=f"v{i}", index=i, azimuth=0.0) for i in range(8)]
    MultiviewResult(id="x", cutout_id="c", views=exactly_eight)  # does not raise


def test_registry_chain_types_line_up():
    chain = [SEGMENT, REMOVE_BG, GENERATE_MULTIVIEW, FIT_TO_LAST]
    for producer, consumer in pairwise(chain):
        assert STEPS[producer].output_model in STEPS[consumer].input_models


def test_latency_jitter_is_deterministic_and_bounded():
    settings = KitSettings(latency_jitter=0.2, seed=1)
    first = _jittered_duration(10.0, settings, "segment", "img", 0)
    second = _jittered_duration(10.0, settings, "segment", "img", 0)
    assert first == second
    assert 8.0 <= first <= 12.0
