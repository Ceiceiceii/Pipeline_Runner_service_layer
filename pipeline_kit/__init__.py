"""Pipeline Runner take-home — starter kit.

Provides the simulated substrate for the exercise: four typed ``async`` pipeline
functions, a cost-metered :class:`GpuPool`, and a :class:`BurstWorkload`
generator. See ``README.md`` for what's provided versus what you build.
"""

from __future__ import annotations

from pipeline_kit.clock import Clock, ManualClock, RealClock
from pipeline_kit.config import KitSettings
from pipeline_kit.errors import (
    PermanentError,
    PipelineError,
    PoolError,
    PoolExhaustedError,
    StepError,
    TransientError,
    WorkerBusyError,
    WorkerNotReadyError,
)
from pipeline_kit.gpu import (
    CostReport,
    GpuPool,
    PoolSnapshot,
    Worker,
    WorkerSnapshot,
    WorkerState,
)
from pipeline_kit.pipelines import (
    STEPS,
    StepInfo,
    fit_to_last,
    generate_multiview,
    remove_bg,
    segment,
)
from pipeline_kit.schemas import Cutout, Image, Mask, Mesh, MultiviewResult, View
from pipeline_kit.workload import BurstWorkload, Request

__version__ = "0.1.0"

__all__ = [
    "STEPS",
    "BurstWorkload",
    "Clock",
    "CostReport",
    "Cutout",
    "GpuPool",
    "Image",
    "KitSettings",
    "ManualClock",
    "Mask",
    "Mesh",
    "MultiviewResult",
    "PermanentError",
    "PipelineError",
    "PoolError",
    "PoolExhaustedError",
    "PoolSnapshot",
    "RealClock",
    "Request",
    "StepError",
    "StepInfo",
    "TransientError",
    "View",
    "Worker",
    "WorkerBusyError",
    "WorkerNotReadyError",
    "WorkerSnapshot",
    "WorkerState",
    "fit_to_last",
    "generate_multiview",
    "remove_bg",
    "segment",
]
