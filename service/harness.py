"""Policy-comparison harness: same workload, different brains, one table.

Replays the identical BurstWorkload schedule through the full service under
each named warm-pool policy on a ManualClock (instant, deterministic, correct
dollars) and prints one row per policy: total $, $/successful job, latency
p50/p95, cold starts paid, utilization, rejections.

Usage:
    python -m service.harness --policies naive,adaptive,always4,eager0 --seed 0
    python -m service.harness --scenario light      # gentler load: shows thrash
    python -m service.harness --scenario chaos      # 30% GPU failure rate

The ``naive`` row is the origin baseline: the kit used with **no service
layer at all**, the way ``pipeline_kit.demo`` uses it — each job leases a
worker, pays its own cold start, cools it afterwards; no queue (a full pool
kills the job), no retries, no idempotency. Everything the service adds is
measured against this row.

Measurement horizon: each run advances until every job is terminal; policies
that scale down keep advancing until the pool is fully cold, so their idle-TTL
"tail rent" is *included*. always-warm-N is cut off at the last job (it would
bill forever), which makes the comparison conservative in its favor.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from pipeline_kit.clock import ManualClock
from pipeline_kit.config import KitSettings
from pipeline_kit.errors import PoolExhaustedError, StepError
from pipeline_kit.gpu import GpuPool, WorkerState
from pipeline_kit.pipelines import STEPS
from pipeline_kit.workload import BurstWorkload, Request
from service.logs import configure_logging
from service.metrics import percentile
from service.models import OverCapacityError
from service.resources import ResourceClass, resource_class
from service.scheduler import (
    AdaptivePolicy,
    AlwaysWarmN,
    WarmPoolPolicy,
    eager_scale_to_zero,
)
from service.service import JobService
from service.simtime import advance_until


def make_policy(name: str, settings: KitSettings) -> WarmPoolPolicy:
    """Build a policy by harness name."""
    factories: dict[str, Callable[[], WarmPoolPolicy]] = {
        "adaptive": AdaptivePolicy,
        "adaptive-warm1": lambda: AdaptivePolicy(min_warm=1),
        "always4": lambda: AlwaysWarmN(settings.max_workers),
        "eager0": lambda: eager_scale_to_zero(5.0),
    }
    if name not in factories:
        raise SystemExit(f"unknown policy {name!r}; choose from {sorted(factories)}")
    return factories[name]()


def make_settings(scenario: str, seed: int) -> KitSettings:
    """Kit settings per scenario."""
    if scenario == "default":
        return KitSettings(seed=seed)
    if scenario == "light":
        # Gentle bursts with long quiets: each spike's backlog drains ~30s
        # into a 60s gap, leaving ~30s of true idleness before the next burst
        # — the regime where eager scale-to-zero thrashes cold starts and the
        # break-even TTL correctly rides the gap warm.
        return KitSettings(
            seed=seed,
            base_rate=0.05,
            burst_rate=1.0,
            quiet_duration_s=60.0,
            burst_duration_s=10.0,
        )
    if scenario == "sparse":
        # Lone jobs ~50s apart: idle gaps shorter than a cold start but far
        # longer than eager0's 5s TTL. The break-even TTL rides the gaps warm
        # (one cold start for the whole run); eager0 re-pays a cold start —
        # in dollars *and* ~45s of latency — for nearly every job.
        return KitSettings(
            seed=seed,
            base_rate=0.02,
            burst_rate=0.0,
            quiet_duration_s=600.0,
            burst_duration_s=0.0,
            n_bursts=1,
        )
    if scenario == "chaos":
        return KitSettings(
            seed=seed, multiview_failure_rate=0.3, fit_to_last_failure_rate=0.3
        )
    raise SystemExit(f"unknown scenario {scenario!r}")


@dataclass
class RunResult:
    """One policy's economics over the identical replayed workload."""

    policy: str
    total_cost: float
    cost_per_success: float
    p50_s: float
    p95_s: float
    cold_starts: int
    utilization: float
    rejected: int
    succeeded: int
    failed: int
    makespan_s: float


async def run_naive(scenario: str, seed: int) -> RunResult:
    """Replay the workload with NO service layer: the origin baseline.

    This is the kit's raw contract, used the way ``pipeline_kit.demo`` uses
    it — per-job lease, warm, run, cool. "Never auto-warms, never queues,
    never sheds, never batches, never scales" means here: every GPU job pays
    its own cold start; a full pool raises ``PoolExhaustedError`` straight
    into the job (counted under "rej"); a transient model failure kills the
    whole chain (no retries).
    """
    settings = make_settings(scenario, seed)
    clock = ManualClock()
    pool = GpuPool.from_settings(settings, clock)

    succeeded = 0
    failed = 0
    shed = 0
    cold_starts = 0
    latencies: list[float] = []

    async def handle(request: Request) -> None:
        nonlocal succeeded, failed, shed, cold_starts
        started = clock.monotonic()
        context: dict[type, object] = {type(request.image): request.image}
        try:
            for name in request.chain:
                info = STEPS[name]
                inputs = [context[model] for model in info.input_models]
                if resource_class(name) is ResourceClass.GPU:
                    with pool.lease() as worker:  # raises PoolExhaustedError
                        if worker.state is WorkerState.COLD:
                            cold_starts += 1
                        try:
                            await worker.warm()  # this job pays the cold start
                            output = await worker.run(
                                info.fn(*inputs, settings=settings, clock=clock)
                            )
                        finally:
                            # Cool even when the step fails: a released-but-
                            # warm worker would bill until the end of the sim
                            # and silently corrupt the baseline's dollars.
                            worker.cool()
                else:
                    output = await info.fn(*inputs, settings=settings, clock=clock)
                context[type(output)] = output
        except PoolExhaustedError:
            shed += 1
        except StepError:
            failed += 1
        else:
            succeeded += 1
            latencies.append(clock.monotonic() - started)

    workload = BurstWorkload(settings)
    drive_task = asyncio.create_task(workload.drive(handle, clock=clock))
    await advance_until(clock, drive_task.done, label="naive settle")
    await drive_task

    # Accounting post-condition: every lease cooled its worker, so the bill
    # is closed. A stranded warm worker would keep billing phantom idle rent
    # for the rest of the (arbitrarily long) measurement horizon.
    if pool.warm_count != 0:
        raise RuntimeError(
            f"naive baseline leaked {pool.warm_count} warm worker(s); "
            "its cost figures would be meaningless"
        )
    cost_report = pool.cost_report()
    return RunResult(
        policy="naive",
        total_cost=cost_report.total_cost,
        cost_per_success=(
            (cost_report.total_cost / succeeded) if succeeded else math.inf
        ),
        p50_s=percentile(latencies, 0.50) or 0.0,
        p95_s=percentile(latencies, 0.95) or 0.0,
        cold_starts=cold_starts,
        utilization=cost_report.utilization,
        rejected=shed,
        succeeded=succeeded,
        failed=failed,
        makespan_s=clock.monotonic(),
    )


async def run_policy(policy_name: str, scenario: str, seed: int) -> RunResult:
    """Replay the workload under one policy and collect its economics."""
    settings = make_settings(scenario, seed)
    clock = ManualClock()
    policy = make_policy(policy_name, settings)
    service = JobService(settings=settings, clock=clock, policy=policy)
    service.start()

    rejected = 0

    async def submit(request: Request) -> str | None:
        nonlocal rejected
        try:
            job, _dedup = service.submit(list(request.chain), request.image)
        except OverCapacityError:
            rejected += 1
            return None
        return job.job_id

    workload = BurstWorkload(settings)
    drive_task = asyncio.create_task(workload.drive(submit, clock=clock))
    await advance_until(
        clock,
        lambda: drive_task.done() and service.all_terminal(),
        label=f"{policy_name} settle",
    )
    await drive_task
    makespan = clock.monotonic()

    # Let scale-down policies pay their idle-TTL tail before we read the bill.
    if math.isfinite(policy.idle_ttl_s) and policy.min_warm == 0:
        await advance_until(
            clock,
            lambda: service.pool.warm_count == 0,
            label=f"{policy_name} cooldown",
        )

    snapshot = service.metrics_snapshot()
    cost = snapshot["gpu"]["total_cost"]
    counters = snapshot["counters"]
    succeeded = counters.get("jobs_succeeded", 0)
    await service.stop()
    return RunResult(
        policy=policy_name,
        total_cost=cost,
        cost_per_success=(cost / succeeded) if succeeded else math.inf,
        p50_s=snapshot["job_latency_s"]["p50"] or 0.0,
        p95_s=snapshot["job_latency_s"]["p95"] or 0.0,
        cold_starts=counters.get("cold_starts", 0),
        utilization=snapshot["gpu"]["utilization"],
        rejected=rejected,
        succeeded=succeeded,
        failed=counters.get("jobs_failed", 0),
        makespan_s=makespan,
    )


def print_table(scenario: str, seed: int, results: list[RunResult]) -> None:
    """Render the comparison table."""
    print(f"\nscenario={scenario} seed={seed}")
    header = (
        f"{'policy':<15} {'total $':>9} {'$/success':>10} {'p50 s':>8} "
        f"{'p95 s':>8} {'cold':>5} {'util':>6} {'rej':>5} {'ok':>5} "
        f"{'fail':>5} {'makespan':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.policy:<15} {r.total_cost:>9.4f} {r.cost_per_success:>10.5f} "
            f"{r.p50_s:>8.1f} {r.p95_s:>8.1f} {r.cold_starts:>5d} "
            f"{r.utilization:>6.2f} {r.rejected:>5d} {r.succeeded:>5d} "
            f"{r.failed:>5d} {r.makespan_s:>9.1f}"
        )


async def main(argv: list[str] | None = None) -> None:
    """Parse args, replay each policy, print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies", default="naive,adaptive,adaptive-warm1,always4,eager0"
    )
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    # Per-event JSON logs would swamp the table; keep the harness quiet.
    # (configure first so JobService's configure_logging can't reset the level)
    configure_logging()
    logging.getLogger("service").setLevel(logging.WARNING)

    results = [
        await run_naive(args.scenario, args.seed)
        if name.strip() == "naive"
        else await run_policy(name.strip(), args.scenario, args.seed)
        for name in args.policies.split(",")
        if name.strip()
    ]
    print_table(args.scenario, args.seed, results)


if __name__ == "__main__":
    asyncio.run(main())
