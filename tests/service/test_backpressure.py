"""Admission control: demand-based backlog bound, typed rejection, CPU bypass."""

from __future__ import annotations

import pytest

from pipeline_kit.schemas import Image
from service.models import JobState, OverCapacityError
from service.scheduler import AdaptivePolicy
from service.service import ServiceConfig
from tests.service.conftest import make_service, settle


async def test_admission_bound_counts_admitted_demand_not_queue_depth(clock):
    """Reject on admitted demand, not visible queue depth.

    Jobs in their CPU prelude hold their reservation: a burst cannot slip
    past a momentarily empty queue.
    """
    service = make_service(
        clock,
        config=ServiceConfig(max_gpu_backlog=2),
        policy=AdaptivePolicy(),
        settings_overrides={
            "multiview_failure_rate": 0.0,
            "fit_to_last_failure_rate": 0.0,
        },
    )
    service.start()
    # Each multiview chain brings one GPU item. The 3rd submit must be
    # rejected IMMEDIATELY — the queue is still empty (all jobs are in their
    # CPU prelude), but the admitted demand already fills the bound.
    admitted = [service.submit("multiview", Image(id=f"img-{i}"))[0] for i in range(2)]
    assert service.scheduler.queue_depth == 0  # nothing enqueued yet...
    with pytest.raises(OverCapacityError) as excinfo:
        service.submit("multiview", Image(id="img-overflow"))
    assert excinfo.value.retry_after_s > 0

    # CPU-only chains never touch the pool and are always admitted.
    cpu_job, _ = service.submit("cutout", Image(id="img-cpu"))
    await settle(clock, until=lambda: cpu_job.is_terminal)
    assert cpu_job.state is JobState.SUCCEEDED

    # Once the admitted work finishes, its reservation frees up.
    await settle(clock, until=lambda: all(j.is_terminal for j in admitted))
    assert all(j.state is JobState.SUCCEEDED for j in admitted)
    late, _ = service.submit("multiview", Image(id="img-late"))
    await settle(clock, until=lambda: late.is_terminal)
    assert late.state is JobState.SUCCEEDED
    await service.stop()


async def test_full_chain_reserves_one_slot_per_gpu_step(clock):
    service = make_service(clock, config=ServiceConfig(max_gpu_backlog=3))
    service.start()
    service.submit("full", Image(id="img-1"))  # reserves 2 items
    service.submit("multiview", Image(id="img-2"))  # reserves 1 item
    with pytest.raises(OverCapacityError):
        service.submit("multiview", Image(id="img-3"))
    await service.stop()


async def test_cancelling_queued_work_frees_admission_immediately(clock):
    service = make_service(clock, config=ServiceConfig(max_gpu_backlog=1))
    service.start()
    first, _ = service.submit("multiview", Image(id="img-1"))
    with pytest.raises(OverCapacityError):
        service.submit("multiview", Image(id="img-2"))
    service.cancel(first.job_id)
    await settle(clock, until=lambda: first.is_terminal)
    assert first.state is JobState.CANCELLED
    # The cancelled job's reservation is gone: admission opens up again.
    replacement, _ = service.submit("multiview", Image(id="img-3"))
    await settle(clock, until=lambda: replacement.is_terminal)
    assert replacement.state is JobState.SUCCEEDED
    await service.stop()


async def test_default_bound_derived_from_wait_promise():
    from pipeline_kit.config import KitSettings

    config = ServiceConfig.from_settings(KitSettings())
    # capacity(4) x promise(300s) / mean GPU item seconds ((8+12)/2) = 120
    assert config.max_gpu_backlog == 120
    # The knob is real: tightening the promise tightens the bound.
    tighter = ServiceConfig.from_settings(KitSettings(), max_acceptable_wait_s=150.0)
    assert tighter.max_gpu_backlog == 60
