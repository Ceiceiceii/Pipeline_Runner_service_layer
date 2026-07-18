"""Runtime configuration for the kit.

Every tunable knob lives in :class:`KitSettings` and is overridable via
``PIPELINE_KIT_*`` environment variables or a ``.env`` file (see ``.env.example``).
The step-name constants and :data:`DEFAULT_TIMINGS` are the single source of
truth shared by the settings defaults and the step registry.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

# Canonical step names, reused by config, the registry, and determinism keys.
SEGMENT = "segment"
REMOVE_BG = "remove_bg"
GENERATE_MULTIVIEW = "generate_multiview"
FIT_TO_LAST = "fit_to_last"

# Nominal (duration_s, failure_rate) per step. The two CPU steps are cheap and
# "always available" (no failures); the two GPU steps are slow and flaky.
DEFAULT_TIMINGS: dict[str, tuple[float, float]] = {
    SEGMENT: (0.5, 0.0),
    REMOVE_BG: (0.8, 0.0),
    GENERATE_MULTIVIEW: (8.0, 0.1),
    FIT_TO_LAST: (12.0, 0.1),
}


class KitSettings(BaseSettings):
    """All tunable knobs for the kit, overridable via ``PIPELINE_KIT_*`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_KIT_",
        env_file=".env",
        extra="ignore",
    )

    # Reproducibility: one seed drives every random decision in the kit.
    seed: int = 0

    # Per-step latency (seconds) and transient failure rate (0.0-1.0).
    segment_duration_s: float = DEFAULT_TIMINGS[SEGMENT][0]
    segment_failure_rate: float = DEFAULT_TIMINGS[SEGMENT][1]
    remove_bg_duration_s: float = DEFAULT_TIMINGS[REMOVE_BG][0]
    remove_bg_failure_rate: float = DEFAULT_TIMINGS[REMOVE_BG][1]
    multiview_duration_s: float = DEFAULT_TIMINGS[GENERATE_MULTIVIEW][0]
    multiview_failure_rate: float = DEFAULT_TIMINGS[GENERATE_MULTIVIEW][1]
    fit_to_last_duration_s: float = DEFAULT_TIMINGS[FIT_TO_LAST][0]
    fit_to_last_failure_rate: float = DEFAULT_TIMINGS[FIT_TO_LAST][1]

    # Fractional latency jitter applied to every step (0.2 => [0.8x, 1.2x]).
    latency_jitter: float = 0.2
    # Probability a step fails permanently (non-retryable) on every attempt.
    permanent_failure_rate: float = 0.0

    # GPU pool / cost model.
    max_workers: int = 4
    cold_start_min_s: float = 30.0
    cold_start_max_s: float = 60.0
    cost_per_second: float = 0.0003

    # Burst workload generator (Poisson arrivals, rate toggles quiet<->burst).
    base_rate: float = 0.2
    burst_rate: float = 5.0
    quiet_duration_s: float = 30.0
    burst_duration_s: float = 10.0
    n_bursts: int = 3

    # 1.0 = real time; higher compresses simulated time on a RealClock.
    time_scale: float = 1.0

    def timing_for(self, step_name: str) -> tuple[float, float]:
        """Return ``(duration_s, failure_rate)`` for ``step_name``."""
        mapping = {
            SEGMENT: (self.segment_duration_s, self.segment_failure_rate),
            REMOVE_BG: (self.remove_bg_duration_s, self.remove_bg_failure_rate),
            GENERATE_MULTIVIEW: (
                self.multiview_duration_s,
                self.multiview_failure_rate,
            ),
            FIT_TO_LAST: (self.fit_to_last_duration_s, self.fit_to_last_failure_rate),
        }
        if step_name not in mapping:
            raise KeyError(f"unknown step: {step_name!r}")
        return mapping[step_name]
