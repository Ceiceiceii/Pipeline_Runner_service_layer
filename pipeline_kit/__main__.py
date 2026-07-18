"""``python -m pipeline_kit`` — preview the bursty workload schedule.

For the end-to-end chain demo, run ``python -m pipeline_kit.demo`` instead.
"""

from __future__ import annotations

from pipeline_kit.config import KitSettings
from pipeline_kit.workload import BurstWorkload, Request


def _print_schedule(schedule: list[Request], bucket_s: float = 10.0) -> None:
    """Print a coarse text histogram so the burstiness is visible."""
    if not schedule:
        print("(empty schedule)")
        return
    span = schedule[-1].t_offset
    buckets: dict[int, int] = {}
    for request in schedule:
        bucket = int(request.t_offset // bucket_s)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    peak = max(buckets.values())
    print(
        f"{len(schedule)} requests over {span:.0f}s "
        f"(peak {peak} per {bucket_s:.0f}s window):"
    )
    for bucket in range(int(span // bucket_s) + 1):
        count = buckets.get(bucket, 0)
        print(f"  t={bucket * bucket_s:5.0f}s | {'#' * count} {count}")


def main() -> None:
    """Print a preview of the default burst schedule."""
    print("Pipeline Runner kit - burst workload preview")
    print("(run `python -m pipeline_kit.demo` for the end-to-end chain demo)\n")
    _print_schedule(BurstWorkload(KitSettings()).schedule())


if __name__ == "__main__":
    main()
