"""Job model, state machines, typed errors, chain validation, and presets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from pipeline_kit.determinism import content_id
from pipeline_kit.pipelines import STEPS
from pipeline_kit.schemas import Image
from pipeline_kit.workload import CUTOUT_CHAIN, FULL_CHAIN, MULTIVIEW_CHAIN

# Presets mirror the chains real traffic asks for, so both consumers can speak
# at intent granularity while raw step lists stay available.
PRESETS: dict[str, tuple[str, ...]] = {
    "full": FULL_CHAIN,
    "multiview": MULTIVIEW_CHAIN,
    "cutout": CUTOUT_CHAIN,
}


class JobState(StrEnum):
    """Lifecycle of a job. Rejected submissions never become jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class StepState(StrEnum):
    """Lifecycle of one step within a chain."""

    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


ErrorKind = Literal["transient_exhausted", "permanent", "cancelled", "over_capacity"]


@dataclass(frozen=True)
class ErrorKindInfo:
    """Everything derivable from an error kind, in one place.

    ``retryable`` drives JobError; ``action``/``reason`` drive the agent
    contract's suggestion. Adding a failure kind is one row here — the runner
    and the agent tools both read this table, so they cannot drift.
    """

    retryable: bool
    action: Literal["resubmit", "give_up", "retry_after"]
    reason: str


ERROR_KINDS: dict[ErrorKind, ErrorKindInfo] = {
    "transient_exhausted": ErrorKindInfo(
        retryable=True,
        action="resubmit",
        reason="transient failures exhausted the retry budget; "
        "a resubmit gets a fresh budget",
    ),
    "permanent": ErrorKindInfo(
        retryable=False,
        action="give_up",
        reason="retrying this input will always fail",
    ),
    "cancelled": ErrorKindInfo(
        retryable=True,
        action="resubmit",
        reason="job was cancelled, not broken",
    ),
    "over_capacity": ErrorKindInfo(
        retryable=True,
        action="retry_after",
        reason="the GPU backlog exceeds the admission bound; wait and resubmit",
    ),
}


class JobError(BaseModel):
    """Machine-actionable failure description surfaced to clients and agents."""

    step_name: str
    attempt: int
    kind: ErrorKind
    retryable: bool
    message: str


class AttemptSpan(BaseModel):
    """One attempt of one step: a span in the job's trace."""

    attempt: int
    t_start: float
    t_end: float | None = None
    outcome: Literal["running", "succeeded", "transient_error", "permanent_error"] = (
        "running"
    )
    worker_id: int | None = None


class StepRun(BaseModel):
    """Execution record for one step of a chain."""

    step_name: str
    state: StepState = StepState.PENDING
    attempts: list[AttemptSpan] = Field(default_factory=list)
    enqueued_at: float | None = None  # GPU steps: when the item entered the queue
    output_id: str | None = None
    memoized: bool = False
    error: JobError | None = None

    @property
    def queue_wait_s(self) -> float | None:
        """Seconds between enqueue and the first attempt starting, if known."""
        if self.enqueued_at is None or not self.attempts:
            return None
        return max(0.0, self.attempts[0].t_start - self.enqueued_at)


class Job(BaseModel):
    """One submitted chain and everything that happened to it."""

    job_id: str
    idempotency_key: str
    chain: tuple[str, ...]
    image: Image
    state: JobState = JobState.QUEUED
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    steps: list[StepRun] = Field(default_factory=list)
    result_ids: dict[str, str] = Field(default_factory=dict)  # step name -> output id
    error: JobError | None = None
    cancel_requested: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def final_result_id(self) -> str | None:
        """Output id of the last completed step of the chain, if any."""
        if not self.chain:
            return None
        return self.result_ids.get(self.chain[-1])


class ServiceError(Exception):
    """Base class for typed errors raised by the service layer."""


class ChainValidationError(ServiceError):
    """The submitted chain cannot run; carries per-problem detail."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class OverCapacityError(ServiceError):
    """The GPU backlog exceeds the admission bound; retry later."""

    def __init__(self, retry_after_s: float, queue_depth: int) -> None:
        super().__init__(
            f"GPU queue is full ({queue_depth} items); retry in ~{retry_after_s:.0f}s"
        )
        self.retry_after_s = retry_after_s
        self.queue_depth = queue_depth


class JobCancelledError(ServiceError):
    """A queued work item was cancelled before it ran."""


def resolve_chain(chain: list[str] | str) -> tuple[str, ...]:
    """Resolve a preset name or explicit step list into a step tuple."""
    if isinstance(chain, str):
        if chain not in PRESETS:
            raise ChainValidationError(
                [f"unknown preset {chain!r}; expected one of {sorted(PRESETS)}"]
            )
        return PRESETS[chain]
    return tuple(chain)


def validate_chain(chain: tuple[str, ...]) -> None:
    """Reject a chain at submit time if it cannot possibly run.

    Walks the chain against the STEPS registry, tracking which output types are
    available (the input Image plus every prior step's output), and names the
    exact step whose inputs cannot be satisfied.
    """
    problems: list[str] = []
    if not chain:
        problems.append("chain is empty")
    unknown = [name for name in chain if name not in STEPS]
    problems.extend(
        f"unknown step {name!r}; expected one of {sorted(STEPS)}" for name in unknown
    )
    if problems:
        raise ChainValidationError(problems)

    available: set[type] = {Image}
    for position, name in enumerate(chain):
        info = STEPS[name]
        problems.extend(
            f"step {position} ({name!r}) needs {model.__name__}, which no "
            f"earlier step produces"
            for model in info.input_models
            if model not in available
        )
        available.add(info.output_model)
    if problems:
        raise ChainValidationError(problems)


def derived_idempotency_key(chain: tuple[str, ...], image: Image) -> str:
    """Default idempotency key: same chain + same image = same job.

    Uses the kit's own canonical content hashing (delimiter-safe, one hashing
    convention across the system) rather than a second ad-hoc scheme.
    """
    return content_id("job", *chain, image.id, image.width, image.height)


class SubmitAck(BaseModel):
    """Acknowledgement for an accepted submission, shared by REST and agent."""

    job_id: str
    state: str
    dedup: bool
    estimated_wait_s: float


def over_capacity_payload(exc: OverCapacityError) -> dict[str, Any]:
    """Build the one over-capacity error shape REST and agent both return."""
    return {
        "error": "over_capacity",
        "retryable": True,
        "retry_after_s": round(exc.retry_after_s, 1),
    }
