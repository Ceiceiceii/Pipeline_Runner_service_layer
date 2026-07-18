# Pipeline Runner — Take-Home Starter Kit

This repo is the **starter kit** for the HILOS Lead Platform Engineer take-home,
*The Pipeline Runner*. It gives you a realistic, fully local simulation of
HILOS's footwear-asset pipeline so you can spend your day on the hard part —
**the service layer** — instead of on scaffolding.

👉 **Read [`CHALLENGE.md`](./CHALLENGE.md) first.** It is the brief: what to
build, how we evaluate, and the deliverables (`DESIGN.md` and a README).

---

## Quickstart

```bash
# Python 3.11+
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q                      # the substrate's own trust tests should pass
python -m pipeline_kit.demo    # run one chain end-to-end; see timings + GPU cost
python -m pipeline_kit         # print a reproducible bursty arrival schedule
```

The kit has two runtime dependencies (`pydantic`, `pydantic-settings`) and no
external services. Everything runs in-process.

---

## What's in the box

```
pipeline_kit/
  schemas.py    Image, Mask, Cutout, View, MultiviewResult, Mesh (typed, content-addressed)
  pipelines.py  segment · remove_bg · generate_multiview · fit_to_last + the STEPS registry
  gpu.py        GpuPool + Worker — a cost-metered GPU fleet (mechanism, no policy)
  workload.py   BurstWorkload — spiky arrival generator + a replay driver
  clock.py      Clock protocol; RealClock (time-scalable) + ManualClock (virtual, for tests)
  config.py     KitSettings — every knob, env-overridable (PIPELINE_KIT_*)
  determinism.py  seeded, cross-platform RNG + content-addressed ids
  demo.py       an illustrative smoke run — NOT a scheduler
```

The four steps match the real pipeline: `segment → remove_bg` are **CPU-bound**
(cheap, fast, always available); `generate_multiview → fit_to_last` are
**GPU-bound** (slow, flaky, and must run on the pool).

---

## Scope: what the kit gives you vs. what you build

The kit deliberately exposes **facts and mechanism** and makes **no scheduling
decision for you**. Designing those decisions is the exercise.

| ✅ Provided by the kit | 🛠️ Yours to build |
| --- | --- |
| Four typed `async` pipeline steps (configurable latency + failure) | Job submission + a job-id lifecycle, and wherever you store it |
| `GpuPool`: cold-start delay, per-second cost, worker lifecycle | The **scheduler**: warm-pool sizing, when to scale to zero, batching |
| A non-blocking saturation signal (`acquire` raises when full) | **Backpressure** policy: queueing, bounded queues, shedding, deadlines, priority |
| Burst workload generator + replay driver | Retry/backoff, idempotency enforcement, dedup |
| One shared `Clock` (real + virtual) | The **chain runner** / step orchestration |
| Deterministic failures + content-addressed output ids | Metric **aggregation**, p50/p95, dashboards, an end-to-end trace |
| Raw cost / utilization / state signals (`cost_report`, `snapshot`) | The **agent-facing** typed contract |
| Step registry: typed I/O + nominal timing | Mapping declared/known resource needs → scheduling |

**Intentionally left to you** (don't look for these — building them is the point):
job submission & job-state model · the chain runner · all retry/backoff,
idempotency & dedup · all backpressure (queue/shed/deadline/fairness/priority) ·
warm-pool sizing, scale-to-zero timing, batching strategy · all metric
aggregation, percentiles, dashboards & traces · the agent-facing contract ·
modeling resource needs and routing on them.

---

## Using the primitives

**Run a chain (CPU steps free; GPU steps on the pool):**

```python
import asyncio
from pipeline_kit import (
    GpuPool, KitSettings, RealClock, Image, segment, remove_bg, generate_multiview,
)

async def main() -> None:
    settings = KitSettings()
    clock = RealClock(settings.time_scale)          # share this clock everywhere
    image = Image(id="photo-1")

    mask = await segment(image, settings=settings, clock=clock)           # CPU: free, no worker
    cutout = await remove_bg(image, mask, settings=settings, clock=clock)

    pool = GpuPool.from_settings(settings, clock)
    with pool.lease() as worker:        # raises PoolExhaustedError if full — backpressure is YOURS
        await worker.warm()             # you decide when to pay the cold start
        views = await worker.run(generate_multiview(cutout, settings=settings, clock=clock))
        # the worker stays warm here — coalesce more work to amortize the cold start
    print(pool.cost_report())

asyncio.run(main())
```

**Drive bursty load against your service:**

```python
from pipeline_kit import BurstWorkload, KitSettings

async def submit(request):              # your service's entry point
    return await my_service.submit_chain(request.chain, request.image)

await BurstWorkload(KitSettings()).drive(submit)   # replays a spiky schedule
```

---

## Guarantees the substrate makes (so your numbers are trustworthy)

- **Cost = warm wall-clock time.** A worker bills `cost_per_second` while
  WARMING, IDLE, *and* BUSY — only COLD is free. Cost is never summed per job, so
  putting more work on an already-warm worker is correctly rewarded (and a failed
  job still bills the time it occupied).
- **The pool is mechanism, never policy.** It never auto-warms, queues, sheds,
  batches, or scales. `acquire()` raises rather than blocks. Every such decision
  is yours.
- **One job per worker.** Each worker is a single GPU slot. "Batching" means one
  job processing many items across a single warm interval — amortizing the cold
  start — which needs no special pool support.
- **Determinism.** A single `seed` drives every random decision. Whether
  `(step, input, attempt)` fails is a pure function of the seed (via `hashlib`,
  so it's identical across OSes and unaffected by async interleaving). Retries
  get an independent chance to succeed. Bursts are materialized up front.
- **Idempotency hook.** Outputs are content-addressed: re-running a step on the
  same input yields an output with the same id.
- **One clock.** The pool's cost meter and the workload driver read the same
  injected `Clock`, so time-compressed runs still bill correctly. `ManualClock`
  makes tests instant and exact.

---

## Configuration

Every knob lives in `KitSettings` and is overridable via `PIPELINE_KIT_*`
environment variables or a `.env` file. Copy [`.env.example`](./.env.example)
and edit. Highlights: per-step `*_duration_s` / `*_failure_rate`,
`latency_jitter`, `permanent_failure_rate`, `max_workers`,
`cold_start_min_s`/`cold_start_max_s`, `cost_per_second`, the burst shape
(`base_rate`, `burst_rate`, `quiet_duration_s`, `burst_duration_s`, `n_bursts`),
`seed`, and `time_scale`.

```python
from pipeline_kit import KitSettings
settings = KitSettings(max_workers=8, cold_start_max_s=45.0, time_scale=30.0)
```

---

## Your submission

See [`CHALLENGE.md`](./CHALLENGE.md) for the full requirements. In short: a
running service, a 1–2 page `DESIGN.md` (architecture, your GPU cost policy, the
one area you took to depth, the platform/ML boundary, and the interaction-loop
appendix), and a README noting any assumptions you made. Good luck — have fun
with it.
