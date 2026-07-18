"""Exception hierarchy raised by the simulated pipeline substrate.

Step failures carry structured context (``step_name``, ``input_id``, ``attempt``)
rather than only a message string, so a chain runner can report *which* step
failed on *which* input at *which* attempt without parsing text. The kit raises
these rich errors; building the localized chain-failure reporting on top of them
is your job.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error raised by the kit."""


class StepError(PipelineError):
    """A pipeline step failed while processing an input."""

    def __init__(
        self,
        message: str,
        *,
        step_name: str,
        input_id: str,
        attempt: int,
    ) -> None:
        super().__init__(message)
        self.step_name = step_name
        self.input_id = input_id
        self.attempt = attempt


class TransientError(StepError):
    """A retryable failure: a flaky model call or transient infra error.

    The intended response is retry-with-backoff; the same ``(step, input)`` has
    an independent chance of succeeding on the next ``attempt``.
    """


class PermanentError(StepError):
    """A non-retryable failure: retrying the same input will always fail."""


class PoolError(PipelineError):
    """Base class for GPU pool and worker errors."""


class PoolExhaustedError(PoolError):
    """Every worker slot is leased; no capacity remains.

    Raised by ``GpuPool.acquire`` instead of blocking — deciding what to do when
    the pool is full (queue, shed, wait with a deadline, prioritize) is a
    backpressure policy, and that policy is yours to write.
    """


class WorkerNotReadyError(PoolError):
    """A job was dispatched to a worker that is not warm and idle.

    Warming is explicit: a cold worker must be ``warm()``-ed (paying the
    cold-start delay) before it can run a job.
    """


class WorkerBusyError(PoolError):
    """A second job was dispatched to a worker already running one.

    Each worker models a single GPU slot and runs one job at a time.
    """
