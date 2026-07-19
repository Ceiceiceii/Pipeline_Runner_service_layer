"""The service's own model of what each step needs to run.

The kit deliberately does not expose a machine-readable resource class — the
step registry carries facts (types, timing) and leaves the modeling to the
service. We declare it here as data, so the runner and scheduler route on a
declared need instead of hard-coding step names anywhere else.
"""

from __future__ import annotations

from enum import StrEnum

from pipeline_kit.config import FIT_TO_LAST, GENERATE_MULTIVIEW, REMOVE_BG, SEGMENT
from pipeline_kit.pipelines import STEPS


class ResourceClass(StrEnum):
    """What a step needs to execute."""

    CPU = "cpu"  # cheap, always available: run inline
    GPU = "gpu"  # scarce, billed while warm: goes through the scheduler


RESOURCE_CLASS: dict[str, ResourceClass] = {
    SEGMENT: ResourceClass.CPU,
    REMOVE_BG: ResourceClass.CPU,
    GENERATE_MULTIVIEW: ResourceClass.GPU,
    FIT_TO_LAST: ResourceClass.GPU,
}

def ensure_all_steps_classified() -> None:
    """Fail closed: an unclassified step must not silently run inline.

    An unmetered, unqueued GPU step would block the event loop for its full
    duration and bypass admission control. Anyone adding a step to the
    registry is forced to declare its resource needs here.
    """
    unclassified = set(STEPS) - set(RESOURCE_CLASS)
    if unclassified:
        raise RuntimeError(
            f"steps registered without a resource class: {sorted(unclassified)}; "
            "declare them in service.resources.RESOURCE_CLASS"
        )


ensure_all_steps_classified()


def resource_class(step_name: str) -> ResourceClass:
    """Return the declared resource class for a step."""
    return RESOURCE_CLASS[step_name]


def needs_gpu(chain: tuple[str, ...]) -> bool:
    """Return True if any step in the chain is GPU-bound."""
    return any(resource_class(name) is ResourceClass.GPU for name in chain)


def gpu_item_count(chain: tuple[str, ...]) -> int:
    """How many GPU work items this chain will enqueue (one per GPU step)."""
    return sum(1 for name in chain if resource_class(name) is ResourceClass.GPU)


def gpu_step_durations() -> list[float]:
    """Nominal durations of every registered GPU step (from the registry)."""
    return [
        info.default_duration_s
        for info in STEPS.values()
        if resource_class(info.name) is ResourceClass.GPU
    ]
