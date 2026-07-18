"""The four mock pipeline functions, plus a metadata registry.

Each function is a pure ``async`` coroutine that sleeps for a (jittered,
deterministic) duration to simulate work and fails at a configurable rate to
simulate flaky models. Outputs are typed and content-addressed.

Resource classes (intentionally *not* encoded as a machine-readable field —
modeling resources is part of the exercise):

* ``segment`` and ``remove_bg`` are **CPU-bound**: cheap, fast, always available.
  Run them anywhere; they cost nothing and need no GPU worker.
* ``generate_multiview`` and ``fit_to_last`` are **GPU-bound**: slow and flaky.
  Execute them on a :class:`~pipeline_kit.gpu.GpuPool` worker via ``worker.run``,
  passing the *same* :class:`~pipeline_kit.clock.Clock` you gave the pool so that
  simulated time and cost stay consistent.

The :data:`STEPS` registry exposes each step's typed input/output models, its
function, and its nominal timing so you can build a *generic* runner (and
validate a chain by type) without hand-coding step specifics.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel

from pipeline_kit.clock import Clock, RealClock
from pipeline_kit.config import (
    DEFAULT_TIMINGS,
    FIT_TO_LAST,
    GENERATE_MULTIVIEW,
    REMOVE_BG,
    SEGMENT,
    KitSettings,
)
from pipeline_kit.determinism import content_id, rng_for
from pipeline_kit.errors import PermanentError, TransientError
from pipeline_kit.schemas import (
    NUM_VIEWS,
    Cutout,
    Image,
    Mask,
    Mesh,
    MultiviewResult,
    View,
)

_DEFAULT_CLOCK: Clock = RealClock()
_MESH_MIN_VERTICES = 20_000
_MESH_MAX_VERTICES = 60_000


def _jittered_duration(
    duration_s: float,
    settings: KitSettings,
    step_name: str,
    input_id: str,
    attempt: int,
) -> float:
    """Apply deterministic latency jitter so observed latency must be measured."""
    if settings.latency_jitter <= 0:
        return duration_s
    spread = settings.latency_jitter
    factor = rng_for(settings.seed, "jitter", step_name, input_id, attempt).uniform(
        1.0 - spread,
        1.0 + spread,
    )
    return max(0.0, duration_s * factor)


def _check_failure(
    step_name: str,
    input_id: str,
    failure_rate: float,
    settings: KitSettings,
    attempt: int,
) -> None:
    """Raise a permanent or transient error per the deterministic failure model.

    The decision is a pure function of ``(seed, step, input, attempt)``, so it is
    identical across runs and unaffected by asyncio interleaving. Because the
    transient draw includes ``attempt``, a retry has an independent chance to
    succeed.
    """
    permanent_roll = rng_for(
        settings.seed, "permanent", step_name, input_id, attempt
    ).random()
    if permanent_roll < settings.permanent_failure_rate:
        raise PermanentError(
            f"{step_name} permanently failed on {input_id}",
            step_name=step_name,
            input_id=input_id,
            attempt=attempt,
        )
    transient_roll = rng_for(
        settings.seed, "transient", step_name, input_id, attempt
    ).random()
    if transient_roll < failure_rate:
        raise TransientError(
            f"{step_name} transiently failed on {input_id} (attempt {attempt})",
            step_name=step_name,
            input_id=input_id,
            attempt=attempt,
        )


async def _do_work(
    step_name: str,
    input_id: str,
    *,
    settings: KitSettings,
    clock: Clock,
    attempt: int,
) -> None:
    """Sleep for the (jittered) step duration, then maybe fail."""
    duration_s, failure_rate = settings.timing_for(step_name)
    await clock.sleep(
        _jittered_duration(duration_s, settings, step_name, input_id, attempt)
    )
    _check_failure(step_name, input_id, failure_rate, settings, attempt)


async def segment(
    image: Image,
    *,
    settings: KitSettings | None = None,
    clock: Clock | None = None,
    attempt: int = 0,
) -> Mask:
    """Segment an image into a mask. **CPU-bound** — cheap, fast, never fails."""
    settings = settings or KitSettings()
    await _do_work(SEGMENT, image.id, settings=settings, clock=clock or _DEFAULT_CLOCK, attempt=attempt)
    return Mask(id=content_id("mask", image.id), image_id=image.id)


async def remove_bg(
    image: Image,
    mask: Mask,
    *,
    settings: KitSettings | None = None,
    clock: Clock | None = None,
    attempt: int = 0,
) -> Cutout:
    """Remove an image's background using its mask. **CPU-bound** — cheap, reliable."""
    settings = settings or KitSettings()
    await _do_work(REMOVE_BG, image.id, settings=settings, clock=clock or _DEFAULT_CLOCK, attempt=attempt)
    return Cutout(
        id=content_id("cutout", image.id, mask.id),
        image_id=image.id,
        mask_id=mask.id,
    )


async def generate_multiview(
    cutout: Cutout,
    *,
    settings: KitSettings | None = None,
    clock: Clock | None = None,
    attempt: int = 0,
) -> MultiviewResult:
    """Generate eight views from a cutout. **GPU-bound** — run on a GpuPool worker."""
    settings = settings or KitSettings()
    await _do_work(GENERATE_MULTIVIEW, cutout.id, settings=settings, clock=clock or _DEFAULT_CLOCK, attempt=attempt)
    views = [
        View(id=content_id("view", cutout.id, index), index=index, azimuth=index * 45.0)
        for index in range(NUM_VIEWS)
    ]
    return MultiviewResult(
        id=content_id("multiview", cutout.id),
        cutout_id=cutout.id,
        views=views,
    )


async def fit_to_last(
    views: MultiviewResult,
    *,
    settings: KitSettings | None = None,
    clock: Clock | None = None,
    attempt: int = 0,
) -> Mesh:
    """Fit views to a 3D last, producing a mesh. **GPU-bound** — run on a GpuPool worker."""
    settings = settings or KitSettings()
    await _do_work(FIT_TO_LAST, views.id, settings=settings, clock=clock or _DEFAULT_CLOCK, attempt=attempt)
    vertices = rng_for(settings.seed, "mesh", views.id).randint(
        _MESH_MIN_VERTICES,
        _MESH_MAX_VERTICES,
    )
    return Mesh(
        id=content_id("mesh", views.id),
        views_id=views.id,
        vertex_count=vertices,
        face_count=vertices * 2,
    )


@dataclass(frozen=True)
class StepInfo:
    """Static metadata describing one pipeline step.

    Deliberately carries *facts* (types, function, nominal timing) but no
    resource class or other field a scheduler could consume directly as a
    routing key — that modeling is left to you.
    """

    name: str
    fn: Callable[..., Awaitable[BaseModel]]
    input_models: tuple[type[BaseModel], ...]
    output_model: type[BaseModel]
    default_duration_s: float
    failure_rate: float


STEPS: dict[str, StepInfo] = {
    SEGMENT: StepInfo(SEGMENT, segment, (Image,), Mask, *DEFAULT_TIMINGS[SEGMENT]),
    REMOVE_BG: StepInfo(
        REMOVE_BG, remove_bg, (Image, Mask), Cutout, *DEFAULT_TIMINGS[REMOVE_BG]
    ),
    GENERATE_MULTIVIEW: StepInfo(
        GENERATE_MULTIVIEW,
        generate_multiview,
        (Cutout,),
        MultiviewResult,
        *DEFAULT_TIMINGS[GENERATE_MULTIVIEW],
    ),
    FIT_TO_LAST: StepInfo(
        FIT_TO_LAST,
        fit_to_last,
        (MultiviewResult,),
        Mesh,
        *DEFAULT_TIMINGS[FIT_TO_LAST],
    ),
}
"""Registry of the four steps, keyed by name. Order reflects the canonical chain."""
