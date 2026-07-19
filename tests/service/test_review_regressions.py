"""Regression tests for the code-review findings (one per confirmed bug)."""

from __future__ import annotations

import pytest

from pipeline_kit.schemas import Image
from service.models import JobState, ServiceError, StepState
from service.scheduler import AdaptivePolicy, AlwaysWarmN
from tests.service.conftest import make_service, settle


async def test_stop_mid_step_leaves_every_job_terminal(clock):
    """service.stop() must never strand a job in a non-terminal state."""
    service = make_service(clock)
    service.start()
    cpu_job, _ = service.submit("cutout", Image(id="img-cpu"))
    gpu_job, _ = service.submit("multiview", Image(id="img-gpu"))
    await clock.advance(0.1)  # both jobs mid-flight (CPU step / queued GPU)
    await service.stop()
    assert cpu_job.is_terminal
    assert gpu_job.is_terminal
    assert cpu_job.state is JobState.CANCELLED
    assert gpu_job.state is JobState.CANCELLED
    assert service.all_terminal()
    assert service.all_terminal(strict=True)  # store agrees with task registry
    # No step left RUNNING/RETRYING, no open attempt span.
    for job in (cpu_job, gpu_job):
        for step in job.steps:
            assert step.state not in (StepState.RUNNING, StepState.RETRYING)
            for span in step.attempts:
                assert span.t_end is not None


async def test_submit_before_start_raises_instead_of_hanging(clock):
    service = make_service(clock)
    with pytest.raises(ServiceError, match="not started"):
        service.submit("multiview", Image(id="img-1"))


async def test_cancelled_job_can_be_resubmitted_under_same_key(clock):
    """Dedup must not trap the caller on a dead cancelled job.

    The agent contract explicitly suggests resubmit for cancelled jobs.
    """
    service = make_service(clock)
    service.start()
    first, _ = service.submit("cutout", Image(id="img-1"))
    await clock.advance(0.1)
    service.cancel(first.job_id)
    await settle(clock, until=lambda: first.is_terminal)
    assert first.state is JobState.CANCELLED

    second, dedup = service.submit("cutout", Image(id="img-1"))
    assert not dedup
    assert second.job_id != first.job_id
    await settle(clock, until=lambda: second.is_terminal)
    assert second.state is JobState.SUCCEEDED
    await service.stop()


async def test_cancel_mid_retry_stops_burning_the_budget(clock):
    """Cancel mid-retry stops the budget.

    A cancelled job must not run its remaining retry attempts on the billed
    worker.
    """
    service = make_service(
        clock,
        policy=AlwaysWarmN(1),
        settings_overrides={"multiview_failure_rate": 1.0},
    )
    service.start()
    job, _ = service.submit("multiview", Image(id="img-1"))
    gpu_step = next(s for s in job.steps if s.step_name == "generate_multiview")
    # Let the first GPU attempt start (cold start ~45s + CPU prelude), then
    # cancel while it is in flight.
    await settle(clock, until=lambda: len(gpu_step.attempts) == 1, step=0.5)
    service.cancel(job.job_id)
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.CANCELLED
    # The in-flight attempt ran out; the other 3 budgeted attempts did not.
    assert len(gpu_step.attempts) == 1
    await service.stop()


async def test_queue_wait_recorded_exactly_once_per_gpu_item(clock):
    service = make_service(
        clock,
        policy=AlwaysWarmN(1),
        settings_overrides={
            "multiview_failure_rate": 0.0,
            "fit_to_last_failure_rate": 0.0,
        },
    )
    service.start()
    job, _ = service.submit("full", Image(id="img-1"))  # 2 GPU items
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.SUCCEEDED
    assert len(service.metrics._queue_waits) == 2
    await service.stop()


async def test_ttl_retirement_race_does_not_strand_new_work(clock):
    """TTL retirement race.

    An item enqueued in the same tick a runner retires must still get a
    runner spawned for it.
    """
    service = make_service(
        clock,
        policy=AdaptivePolicy(target_drain_s=60.0, idle_ttl_s=10.0),
        settings_overrides={
            "multiview_failure_rate": 0.0,
            "fit_to_last_failure_rate": 0.0,
        },
    )
    service.start()
    first, _ = service.submit("multiview", Image(id="img-1"))
    await settle(clock, until=lambda: first.is_terminal)
    # Let the idle TTL expire so the runner retires.
    await settle(clock, until=lambda: service.pool.warm_count == 0, max_advance=30)
    # Immediately submit again: the done-callback lag window.
    second, _ = service.submit("multiview", Image(id="img-2"))
    await settle(clock, until=lambda: second.is_terminal)
    assert second.state is JobState.SUCCEEDED
    await service.stop()


async def test_agent_tool_bad_payload_returns_structured_error(clock):
    from service.agent import AgentTools

    service = make_service(clock)
    service.start()
    tools = AgentTools(service)
    result = await tools.call("get_job", {})  # job_id missing
    assert result["error"] == "invalid_arguments"
    assert any(item["field"] == "job_id" for item in result["detail"])

    result = await tools.call("submit_chain", {"preset": "cutout"})  # no image_id
    assert result["error"] == "invalid_arguments"
    await service.stop()


async def test_naive_baseline_cools_workers_even_on_failure():
    """Naive baseline must not strand billing workers on step failure.

    A leaked warm worker would flatter the service in every failure scenario.
    """
    from service.harness import run_naive

    # chaos produces naive step failures; run_naive itself raises if any
    # lease leaked a warm (still-billing) worker after the run.
    result = await run_naive("chaos", seed=0)
    assert result.failed > 0


async def test_unclassified_step_fails_closed():
    """Fail closed on unclassified steps.

    A step registered without a resource class must be a startup error, not
    a silent inline run.
    """
    from pipeline_kit.pipelines import STEPS
    from service.resources import ensure_all_steps_classified

    STEPS["brand_new_step"] = STEPS["segment"]
    try:
        with pytest.raises(RuntimeError, match="brand_new_step"):
            ensure_all_steps_classified()
    finally:
        del STEPS["brand_new_step"]
    ensure_all_steps_classified()  # clean registry passes again
