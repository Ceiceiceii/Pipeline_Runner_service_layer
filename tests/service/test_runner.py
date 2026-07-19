"""ChainRunner behavior through the service facade, on virtual time."""

from __future__ import annotations

from pipeline_kit.schemas import Image
from service.models import JobState, StepState
from tests.service.conftest import make_service, settle


async def test_cpu_only_chain_succeeds_without_touching_the_pool(clock):
    service = make_service(clock)
    service.start()
    job, dedup = service.submit("cutout", Image(id="img-1"))
    assert not dedup
    assert job.state is JobState.QUEUED or job.state is JobState.RUNNING
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.SUCCEEDED
    # segment 0.5s + remove_bg 0.8s, no queue, no GPU
    assert job.finished_at is not None
    assert job.finished_at - job.created_at == 1.3
    assert set(job.result_ids) == {"segment", "remove_bg"}
    assert job.final_result_id == job.result_ids["remove_bg"]
    assert service.pool.snapshot().cost.total_cost == 0.0
    await service.stop()


async def test_full_chain_succeeds_with_provenance(clock):
    service = make_service(
        clock,
        settings_overrides={
            "multiview_failure_rate": 0.0,
            "fit_to_last_failure_rate": 0.0,
        },
    )
    service.start()
    job, _ = service.submit("full", Image(id="img-1"))
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.SUCCEEDED
    assert [s.state for s in job.steps] == [StepState.SUCCEEDED] * 4
    assert set(job.result_ids) == {
        "segment", "remove_bg", "generate_multiview", "fit_to_last",
    }
    # GPU work happened, so money was spent and exactly one worker warmed.
    assert service.pool.snapshot().cost.total_cost > 0
    assert service.metrics.counters["cold_starts"] == 1
    await service.stop()


async def test_mid_chain_failure_is_localized_with_prior_outputs_intact(clock):
    service = make_service(
        clock,
        settings_overrides={
            "multiview_failure_rate": 1.0,  # always transient-fails
            "fit_to_last_failure_rate": 0.0,
        },
    )
    service.start()
    job, _ = service.submit("full", Image(id="img-1"))
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.FAILED
    assert job.error is not None
    assert job.error.step_name == "generate_multiview"
    assert job.error.kind == "transient_exhausted"
    assert job.error.retryable is True
    by_name = {s.step_name: s for s in job.steps}
    assert by_name["segment"].state is StepState.SUCCEEDED
    assert by_name["remove_bg"].state is StepState.SUCCEEDED
    assert by_name["remove_bg"].output_id is not None
    assert by_name["generate_multiview"].state is StepState.FAILED
    assert len(by_name["generate_multiview"].attempts) == 4  # full retry budget
    assert by_name["fit_to_last"].state is StepState.PENDING  # never started
    await service.stop()


async def test_permanent_failure_short_circuits_and_is_not_retryable(clock):
    service = make_service(
        clock, settings_overrides={"permanent_failure_rate": 1.0}
    )
    service.start()
    job, _ = service.submit("cutout", Image(id="img-1"))
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.FAILED
    assert job.error is not None
    assert job.error.kind == "permanent"
    assert job.error.retryable is False
    assert len(job.steps[0].attempts) == 1  # no retries burned
    await service.stop()


async def test_memoization_skips_completed_steps_on_overlapping_chain(clock):
    service = make_service(clock)
    service.start()
    first, _ = service.submit("cutout", Image(id="img-1"))
    await settle(clock, until=lambda: first.is_terminal)
    assert first.state is JobState.SUCCEEDED

    second, dedup = service.submit(
        ["segment", "remove_bg"], Image(id="img-1"), idempotency_key="explicit"
    )
    assert not dedup  # different idempotency key: a new job...
    await settle(clock, until=lambda: second.is_terminal)
    assert second.state is JobState.SUCCEEDED
    assert all(s.memoized for s in second.steps)  # ...but zero re-execution
    assert second.result_ids == first.result_ids  # content-addressed proof
    await service.stop()


async def test_idempotent_resubmit_returns_same_job_without_rerunning(clock):
    service = make_service(clock)
    service.start()
    first, dedup_first = service.submit("cutout", Image(id="img-1"))
    second, dedup_second = service.submit("cutout", Image(id="img-1"))
    assert not dedup_first
    assert dedup_second
    assert first.job_id == second.job_id
    await settle(clock, until=lambda: first.is_terminal)
    assert service.metrics.counters["submits_deduped"] == 1
    # Only one job's worth of attempts ran.
    assert service.metrics.counters["step_attempts"] == 2
    await service.stop()


async def test_cancel_stops_before_next_step(clock):
    service = make_service(clock)
    service.start()
    job, _ = service.submit("cutout", Image(id="img-1"))
    await clock.advance(0.1)  # inside segment's 0.5s sleep
    service.cancel(job.job_id)
    await settle(clock, until=lambda: job.is_terminal)
    assert job.state is JobState.CANCELLED
    by_name = {s.step_name: s for s in job.steps}
    assert by_name["remove_bg"].state is StepState.PENDING
    await service.stop()
