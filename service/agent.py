"""Agent-facing tool contract: four coarse operations over the JobService.

Deliberately not a CRUD dump. An LLM agent needs: submit at intent granularity
(presets matching real traffic, raw steps still available), progress + a
machine-actionable failure with a suggested next action, best-effort cancel,
and — the piece most agent APIs miss — enough system state to decide *whether
to submit right now*.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from pipeline_kit.schemas import Image
from service.models import (
    ERROR_KINDS,
    ChainValidationError,
    Job,
    JobState,
    OverCapacityError,
    SubmitAck,
    over_capacity_payload,
)
from service.service import JobService


class SubmitChainInput(BaseModel):
    """Submit a pipeline chain as an async job."""

    preset: Literal["full", "multiview", "cutout"] | None = Field(
        default=None,
        description="Named chain: full=segment→remove_bg→multiview→fit_to_last, "
        "multiview stops before the mesh, cutout is CPU-only.",
    )
    steps: list[str] | None = Field(
        default=None,
        description="Explicit step list; use instead of preset for custom chains.",
    )
    image_id: str = Field(description="Id of the input image.")
    idempotency_key: str | None = Field(
        default=None,
        description="Optional; resubmitting the same key never double-runs.",
    )


class GetJobInput(BaseModel):
    """Fetch a job's status, per-step progress, results, and failure detail."""

    job_id: str


class CancelJobInput(BaseModel):
    """Cancel a job, best-effort.

    Queued work is removed; an in-flight GPU attempt runs to completion
    before the job stops.
    """

    job_id: str


class GetSystemStatusInput(BaseModel):
    """Report queue depth, expected wait, capacity, and burn rate.

    Call before submitting bulk work to decide whether to submit now or
    back off.
    """


TOOLS: list[dict[str, Any]] = [
    {
        "name": "submit_chain",
        "description": SubmitChainInput.__doc__,
        "input_schema": SubmitChainInput.model_json_schema(),
    },
    {
        "name": "get_job",
        "description": GetJobInput.__doc__,
        "input_schema": GetJobInput.model_json_schema(),
    },
    {
        "name": "cancel_job",
        "description": CancelJobInput.__doc__,
        "input_schema": CancelJobInput.model_json_schema(),
    },
    {
        "name": "get_system_status",
        "description": GetSystemStatusInput.__doc__,
        "input_schema": GetSystemStatusInput.model_json_schema(),
    },
]


def _suggestion(job: Job) -> dict[str, Any] | None:
    """Turn a failure into the action an agent should take next.

    Pure table lookup: the kind→action mapping lives in one place
    (models.ERROR_KINDS) so a new failure kind cannot silently fall through
    to a wrong suggestion here.
    """
    if job.error is None:
        return None
    info = ERROR_KINDS[job.error.kind]
    return {"action": info.action, "reason": info.reason}


def job_payload(job: Job) -> dict[str, Any]:
    """Render the agent-visible view of one job."""
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "dedup_key": job.idempotency_key,
        "chain": list(job.chain),
        "steps": [
            {
                "step": s.step_name,
                "state": s.state.value,
                "attempts": len(s.attempts),
                "memoized": s.memoized,
                "output_id": s.output_id,
            }
            for s in job.steps
        ],
        "result_ids": job.result_ids,
        "final_result_id": job.final_result_id,
        "error": job.error.model_dump() if job.error else None,
        "suggestion": _suggestion(job) if job.state is not JobState.SUCCEEDED else None,
    }


class AgentTools:
    """Dispatches tool calls onto a JobService."""

    def __init__(self, service: JobService) -> None:
        self._service = service

    async def call(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {
                "error": f"unknown tool {name!r}",
                "tools": [t["name"] for t in TOOLS],
            }
        try:
            result: dict[str, Any] = await handler(payload)
        except ValidationError as exc:
            # A malformed payload must come back machine-actionable like
            # every other tool error, never as a bare 500.
            return {
                "error": "invalid_arguments",
                "detail": [
                    {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                    for e in exc.errors()
                ],
            }
        return result

    async def _tool_submit_chain(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = SubmitChainInput.model_validate(payload)
        chain: list[str] | str
        if params.preset is not None:
            chain = params.preset
        elif params.steps:
            chain = params.steps
        else:
            return {"error": "provide either preset or steps"}
        try:
            job, dedup = self._service.submit(
                chain,
                Image(id=params.image_id),
                idempotency_key=params.idempotency_key,
            )
        except ChainValidationError as exc:
            return {"error": "invalid_chain", "problems": exc.problems}
        except OverCapacityError as exc:
            return over_capacity_payload(exc)
        return SubmitAck(
            job_id=job.job_id,
            state=job.state.value,
            dedup=dedup,
            estimated_wait_s=round(self._service.estimated_wait_s(), 1),
        ).model_dump()

    async def _tool_get_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = GetJobInput.model_validate(payload)
        job = self._service.get_job(params.job_id)
        if job is None:
            return {"error": "not_found", "job_id": params.job_id}
        return job_payload(job)

    async def _tool_cancel_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = CancelJobInput.model_validate(payload)
        job = self._service.cancel(params.job_id)
        if job is None:
            return {"error": "not_found", "job_id": params.job_id}
        return {
            "job_id": job.job_id,
            "state": job.state.value,
            "cancel_requested": job.cancel_requested,
            "note": "best-effort: an in-flight GPU attempt runs to completion",
        }

    async def _tool_get_system_status(self, _payload: dict[str, Any]) -> dict[str, Any]:
        return self._service.system_status()
