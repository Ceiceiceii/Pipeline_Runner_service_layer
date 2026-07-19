"""FastAPI surface: typed REST for the frontend, tool endpoints for the agent.

Run with: uvicorn --factory service.api:create_app
(time compression via PIPELINE_KIT_TIME_SCALE, e.g. =20 for a fast demo).

Deliberately factory-only: constructing the service (and its GpuPool) at
import time would give every importer — and every uvicorn worker process —
its own pool and admission bound, silently multiplying capacity. One process
= one pool; multi-process needs the external store/broker named in
DESIGN.md's skips.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pipeline_kit.config import KitSettings
from pipeline_kit.schemas import Image
from service.agent import TOOLS, AgentTools, job_payload
from service.models import (
    ChainValidationError,
    Job,
    JobState,
    OverCapacityError,
    SubmitAck,
    over_capacity_payload,
)
from service.service import JobService

_POLL_INTERVAL_REAL_S = 0.25  # SSE poll cadence (real seconds, not clock time)
_DEFAULT_PAGE_LIMIT = 100


class SubmitRequest(BaseModel):
    """POST /jobs body: a preset name or an explicit step list."""

    chain: list[str] | str = Field(
        description="Preset ('full' | 'multiview' | 'cutout') or step names."
    )
    image: Image
    idempotency_key: str | None = None


def _change_key(job: Job) -> tuple[object, ...]:
    """Cheap fingerprint of a job's observable progress (for SSE polling)."""
    return (
        job.state,
        tuple((step.state, len(step.attempts)) for step in job.steps),
    )


def create_app(service: JobService | None = None) -> FastAPI:  # noqa: C901, PLR0915 - a route table, not branching logic
    """Build the app; a service can be injected for tests."""
    svc = service or JobService(settings=KitSettings())
    tools = AgentTools(svc)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        svc.start()
        yield
        await svc.stop()

    app = FastAPI(title="Pipeline Runner", lifespan=lifespan)
    app.state.service = svc

    def _get_job_or_404(job_id: str) -> Job:
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return job

    @app.post("/jobs", status_code=202, response_model=SubmitAck)
    async def submit(body: SubmitRequest) -> SubmitAck:
        try:
            job, dedup = svc.submit(
                body.chain, body.image, idempotency_key=body.idempotency_key
            )
        except ChainValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.problems) from exc
        except OverCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail=over_capacity_payload(exc),
                headers={"Retry-After": str(int(exc.retry_after_s) + 1)},
            ) from exc
        return SubmitAck(
            job_id=job.job_id,
            state=job.state.value,
            dedup=dedup,
            estimated_wait_s=round(svc.estimated_wait_s(), 1),
        )

    @app.get("/jobs")
    async def list_jobs(
        limit: Annotated[int, Query(ge=1, le=1000)] = _DEFAULT_PAGE_LIMIT,
        state: JobState | None = None,
    ) -> list[dict[str, Any]]:
        """Most recent jobs first, paginated — the history grows unboundedly."""
        jobs = svc.list_jobs()
        if state is not None:
            jobs = [job for job in jobs if job.state is state]
        return [job_payload(job) for job in jobs[-limit:][::-1]]

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        return job_payload(_get_job_or_404(job_id))

    @app.get("/jobs/{job_id}/trace")
    async def get_trace(job_id: str) -> dict[str, Any]:
        _get_job_or_404(job_id)
        trace = svc.trace(job_id)
        if trace is None:  # pragma: no cover - job existed a line ago
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return trace

    @app.get("/jobs/{job_id}/events")
    async def stream_events(job_id: str) -> StreamingResponse:
        """SSE: emits the job payload on every state change until terminal."""
        _get_job_or_404(job_id)

        async def events() -> AsyncIterator[str]:
            last_key: tuple[object, ...] | None = None
            while True:
                job = svc.get_job(job_id)
                if job is None:
                    break
                # Serialize only when the cheap fingerprint changed — not on
                # every poll of every connected client.
                key = _change_key(job)
                if key != last_key:
                    last_key = key
                    yield f"data: {json.dumps(job_payload(job))}\n\n"
                if job.is_terminal:
                    break
                await asyncio.sleep(_POLL_INTERVAL_REAL_S)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/jobs/{job_id}/cancel")
    async def cancel(job_id: str) -> dict[str, Any]:
        job = svc.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return job_payload(job)

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return svc.metrics_snapshot()

    @app.get("/system")
    async def system() -> dict[str, Any]:
        return svc.system_status()

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/agent/tools")
    async def agent_tools() -> list[dict[str, Any]]:
        return TOOLS

    @app.post("/agent/tools/{tool_name}")
    async def agent_call(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await tools.call(tool_name, payload)

    return app
