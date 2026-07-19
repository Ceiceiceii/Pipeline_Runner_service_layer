"""Job storage.

``JobStore`` is a Protocol so the persistence choice is a seam, not a rewrite:
the in-memory implementation is honest for a single-process simulation, and a
Postgres/Redis-backed store slots in behind the same five methods. Building one
is a declared skip (see DESIGN.md).
"""

from __future__ import annotations

from typing import Protocol

from service.models import Job


class JobStore(Protocol):
    """Minimal storage contract the service depends on."""

    def put(self, job: Job) -> None: ...

    def get(self, job_id: str) -> Job | None: ...

    def get_by_idempotency_key(self, key: str) -> Job | None: ...

    def list_jobs(self) -> list[Job]: ...


class InMemoryJobStore:
    """Dict-backed store: the only implementation this exercise needs."""

    def __init__(self) -> None:
        self._by_id: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}

    def put(self, job: Job) -> None:
        self._by_id[job.job_id] = job
        self._by_key[job.idempotency_key] = job.job_id

    def get(self, job_id: str) -> Job | None:
        return self._by_id.get(job_id)

    def get_by_idempotency_key(self, key: str) -> Job | None:
        job_id = self._by_key.get(key)
        return self._by_id.get(job_id) if job_id is not None else None

    def list_jobs(self) -> list[Job]:
        return list(self._by_id.values())
