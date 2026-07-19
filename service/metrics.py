"""Metric aggregation: counters, latency percentiles, and trace assembly.

Latencies live in bounded reservoirs — at this scale (hundreds of jobs) exact
percentiles over a capped window are simpler and more trustworthy than a
streaming sketch; that swap is a declared skip.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable
from typing import Any

from pipeline_kit.gpu import GpuPool, PoolSnapshot
from service.models import Job, JobState, StepState
from service.resources import ResourceClass, resource_class

_RESERVOIR = 2048


def percentile(values: list[float], q: float) -> float | None:
    """Exact q-quantile (0 < q <= 1) of a small sample, or None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    """p50/p95 from one sort of the sample (snapshots are polled; sort once)."""
    ordered = sorted(values)
    if not ordered:
        return {"p50": None, "p95": None}
    return {
        "p50": ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)],
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


def gpu_economics(snapshot: PoolSnapshot) -> dict[str, Any]:
    """Project pool economics into the one shape every payload shares.

    /metrics and /system must not hand-roll their own key names for the same
    CostReport fields.
    """
    return {
        "warm_workers": snapshot.warm_count,
        "busy_workers": snapshot.busy_count,
        "utilization": snapshot.cost.utilization,
        "total_cost": snapshot.cost.total_cost,
        "burn_rate_per_second": snapshot.cost.burn_rate_per_second,
    }


class MetricsRegistry:
    """Counters + latency reservoirs, merged with the pool's own cost truth."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self._job_latencies: deque[float] = deque(maxlen=_RESERVOIR)
        self._step_latencies: dict[str, deque[float]] = {}
        self._queue_waits: deque[float] = deque(maxlen=_RESERVOIR)

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def observe_queue_wait(self, seconds: float) -> None:
        self._queue_waits.append(seconds)

    def observe_step(self, step_name: str, duration_s: float) -> None:
        self._step_latencies.setdefault(step_name, deque(maxlen=_RESERVOIR)).append(
            duration_s
        )

    def observe_job(self, job: Job) -> None:
        """Record a job reaching a terminal state.

        Queue waits are NOT re-recorded here: the scheduler records each
        item's wait once, live, when the item starts — recording per step
        again at job completion would double-weight completed jobs.
        """
        self.inc(f"jobs_{job.state.value}")
        if job.state is JobState.SUCCEEDED and job.finished_at is not None:
            self._job_latencies.append(job.finished_at - job.created_at)
        for step in job.steps:
            if step.state is StepState.SUCCEEDED and not step.memoized and step.attempts:
                first, last = step.attempts[0], step.attempts[-1]
                if last.t_end is not None:
                    self.observe_step(step.step_name, last.t_end - first.t_start)

    def snapshot(self, pool: GpuPool, **extra: Any) -> dict[str, Any]:
        """One consistent metrics payload: service counters + pool economics."""
        pool_snapshot = pool.snapshot()
        attempts = self.counters["step_attempts"]
        failures = self.counters["step_attempt_failures"]
        return {
            "counters": dict(self.counters),
            "job_latency_s": {
                **_quantiles(self._job_latencies),
                "count": len(self._job_latencies),
            },
            "step_latency_s": {
                name: _quantiles(values)
                for name, values in self._step_latencies.items()
            },
            "queue_wait_s": _quantiles(self._queue_waits),
            "failure_rate": (failures / attempts) if attempts else 0.0,
            "gpu": gpu_economics(pool_snapshot),
            **extra,
        }


def build_trace(job: Job, cost_per_second: float) -> dict[str, Any]:
    """Render a job's StepRun spans as an end-to-end timeline with cost.

    A GPU step's attributed cost covers the interval its worker was held —
    first attempt start to last attempt end, backoff included, because the
    worker bills throughout. Warm-idle time outside any job is deliberately
    *not* smeared across jobs; it shows up at pool level as unattributed cost.
    """
    steps: list[dict[str, Any]] = []
    for step in job.steps:
        held_s = 0.0
        if step.attempts and step.attempts[-1].t_end is not None:
            held_s = step.attempts[-1].t_end - step.attempts[0].t_start
        is_gpu = resource_class(step.step_name) is ResourceClass.GPU
        steps.append(
            {
                "step": step.step_name,
                "state": step.state.value,
                "memoized": step.memoized,
                "queue_wait_s": step.queue_wait_s,
                "worker_id": next(
                    (a.worker_id for a in step.attempts if a.worker_id is not None),
                    None,
                ),
                "attempts": [
                    {
                        "attempt": span.attempt,
                        "t_start": span.t_start,
                        "t_end": span.t_end,
                        "outcome": span.outcome,
                    }
                    for span in step.attempts
                ],
                "attributed_cost": held_s * cost_per_second if is_gpu else 0.0,
                "output_id": step.output_id,
                "error": step.error.model_dump() if step.error else None,
            }
        )
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "duration_s": (
            (job.finished_at - job.created_at) if job.finished_at is not None else None
        ),
        "steps": steps,
        "total_attributed_cost": sum(s["attributed_cost"] for s in steps),
    }
