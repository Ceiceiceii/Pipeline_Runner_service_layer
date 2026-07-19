"""Service layer for the pipeline runner take-home.

The kit (``pipeline_kit``) provides mechanism — typed steps, a cost-metered GPU
pool, a bursty workload. This package provides the policy and the product
surface: job lifecycle, cost-aware GPU scheduling, backpressure, retries,
idempotency, observability, and an agent-drivable contract.
"""

from __future__ import annotations
