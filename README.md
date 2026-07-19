# Pipeline Runner — Take-Home Starter Kit

This repo is the **starter kit** for the HILOS Lead Platform Engineer take-home,
*The Pipeline Runner*, plus the submitted **service layer** built on top of it.

👉 **Read [`CHALLENGE.md`](./CHALLENGE.md) first.** It is the brief: what to
build, how we evaluate, and the deliverables (`DESIGN.md` and a README).

---

## The service

The service layer lives in [`service/`](./service); the design and its defense
— architecture, the GPU cost policy and its math, the deep-dive, tradeoffs,
deliberate skips, and the platform/ML boundary — are in
[`DESIGN.md`](./DESIGN.md). The kit is untouched.

**Endpoints:** `POST /jobs` (202 + job id; 422 invalid chain; 503 + Retry-After
over capacity) · `GET /jobs` (paginated, filterable) · `GET /jobs/{id}` ·
`GET /jobs/{id}/trace` · `GET /jobs/{id}/events` (SSE) ·
`POST /jobs/{id}/cancel` (best-effort) · `GET /metrics` · `GET /system` ·
`GET|POST /agent/tools[/{name}]` · `GET /healthz`.

**Assumptions / declared limits:** in-memory job store behind a `JobStore`
Protocol (jobs die with the process; resubmit is idempotent) · single process,
one event loop (the API is factory-only; `--workers N` would give each process
its own pool) · no auth or per-client fairness · cancellation is best-effort —
an in-flight GPU attempt finishes its current try, then stops; a cancelled
job's chain+image may be resubmitted and gets a new job · queued-vs-warm knobs
(`ServiceConfig`, `AdaptivePolicy`) default to values derived in `DESIGN.md`,
not tuned to the burst schedule. Full skip list with reasons: `DESIGN.md`.

---

## Setup

```bash
# Python 3.11+
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,api]"
```

Two runtime kit dependencies (`pydantic`, `pydantic-settings`); the API extra
adds `fastapi`/`uvicorn`/`httpx`. No external services — everything runs
in-process.

## Verify the submission

**If you only run three commands, make them:** `pytest -q`,
`python -m service.harness`, and the submit → cancel → resubmit curl sequence
below — together they cover the test suite, the economics evidence, and the
trickiest lifecycle path.

### 1. Tests and static checks

```bash
pytest -q            # 19 kit trust tests + 61 service tests (~1s: virtual clock, no real waiting)
ruff check service/ tests/service/    # lint, select-ALL profile (config in pyproject)
mypy service/                         # strict mode (config in pyproject)
```

`tests/service/test_review_regressions.py` pins the lifecycle, admission, and
shutdown edge cases (cancellation mid-step, bursts during the CPU prelude,
stop-with-queued-work, and similar); `tests/service/test_e2e.py` replays a
full burst schedule through the whole service.

### 2. Economics harness (the evidence behind `DESIGN.md`)

```bash
python -m service.harness                      # default saturated bursts (naive baseline first)
python -m service.harness --scenario sparse    # idle-dominated: where the warm-pool policies split
python -m service.harness --scenario chaos     # 30% GPU failure: retry economics
for s in 1 2 3; do python -m service.harness --seed $s; done   # ranking stability
```

Every number in `DESIGN.md`'s table reproduces from these commands
(instant — virtual clock).

### 3. Live API

```bash
# Time compression: 50x makes a ~45s cold start take ~1s of real time
PIPELINE_KIT_TIME_SCALE=50 uvicorn --factory service.api:create_app
```

In a second terminal:

```bash
# Submit a full chain (202 with a job_id immediately)
curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"chain": "full", "image": {"id": "photo-1"}}'

# Status / trace / stream (fill in the job_id)
curl localhost:8000/jobs/<job_id>              # per-step progress, attempts, actionable errors
curl localhost:8000/jobs/<job_id>/trace        # timeline with queue wait + $ attributed per step
curl -N localhost:8000/jobs/<job_id>/events    # SSE stream of state changes until terminal

# Cancel, then resubmit the same chain+image — creates a NEW job
# (a cancelled job releases its capacity and does not trap the idempotency key)
curl -X POST localhost:8000/jobs/<job_id>/cancel
curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"chain": "full", "image": {"id": "photo-1"}}'

# Error paths
curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"chain": ["segment", "generate_multiview"], "image": {"id": "x"}}'
     # ^ 422 naming the incompatible step pair, at submit — not mid-run
curl -i -X POST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"chain": "full", "image": {"id": "y"}}'
     # ^ repeat quickly (~60x with distinct image ids) to see the 503 + Retry-After shed path

# Observability + listing
curl localhost:8000/metrics                    # p50/p95, queue depth, utilization, live $
curl localhost:8000/system                     # should-I-submit-now signal (backlog vs capacity)
curl "localhost:8000/jobs?limit=10&state=succeeded"

# Agent contract
curl localhost:8000/agent/tools                # the 4 tool schemas
curl -X POST localhost:8000/agent/tools/get_system_status -d '{}' -H 'content-type: application/json'
curl -X POST localhost:8000/agent/tools/submit_chain -H 'content-type: application/json' \
     -d '{"preset": "cutout", "image_id": "img-agent-1"}'
curl -X POST localhost:8000/agent/tools/get_job -d '{}' -H 'content-type: application/json'
     # ^ malformed on purpose: returns a typed {"error":"invalid_arguments",...}, not a 500
```

### 4. Kit-only sanity (unchanged substrate)

```bash
python -m pipeline_kit.demo    # one chain end-to-end; timings + GPU cost
python -m pipeline_kit         # print the reproducible bursty arrival schedule
```

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

service/
  models.py     Job/StepRun/AttemptSpan, typed errors, chain validation, presets
  resources.py  CPU/GPU step classification (fail-closed: unclassified steps error at import)
  store.py      JobStore Protocol + in-memory impl (the "swap in Postgres" seam)
  retry.py      backoff policy + the shared attempt loop
  runner.py     generic chain executor (registry-driven, no per-step code)
  scheduler.py  bounded queue, admission control, worker-runners, warm-pool policies
  service.py    facade: submit/dedup/cancel, wiring, lifecycles
  simtime.py    shared virtual-clock advance helper (harness + tests)
  metrics.py    counters, p50/p95 reservoirs, trace assembly
  api.py        FastAPI: REST + SSE (factory-only)
  agent.py      the 4 agent tools + failure→suggestion mapping
  harness.py    replay the same workload under naive/adaptive/always4/eager0
tests/service/  61 tests on the virtual clock (instant, deterministic)
```

The four steps match the real pipeline: `segment → remove_bg` are **CPU-bound**
(cheap, fast, always available); `generate_multiview → fit_to_last` are
**GPU-bound** (slow, flaky, and must run on the pool).

---

## Scope: what the kit gives you vs. what the service builds

The kit deliberately exposes **facts and mechanism** and makes **no scheduling
decision**. The service layer supplies every decision:

| ✅ Provided by the kit | 🛠️ Built in `service/` |
| --- | --- |
| Four typed `async` pipeline steps (configurable latency + failure) | Job submission, job-id lifecycle, in-memory store behind a Protocol |
| `GpuPool`: cold-start delay, per-second cost, worker lifecycle | The **scheduler**: long-lived worker-runners, warm-pool sizing, scale-to-zero |
| A non-blocking saturation signal (`acquire` raises when full) | **Backpressure**: demand-based admission with a typed 503 + Retry-After |
| Burst workload generator + replay driver | Retry/backoff, idempotency, dedup, step memoization |
| One shared `Clock` (real + virtual) | The **chain runner** / step orchestration, cancellation lifecycle |
| Deterministic failures + content-addressed output ids | Metric **aggregation**, p50/p95, live $, end-to-end trace |
| Raw cost / utilization / state signals | The **agent-facing** typed contract (4 coarse tools) |
| Step registry: typed I/O + nominal timing | `ResourceClass` declaration → scheduling, validated at import |

---

## Guarantees the substrate makes (so the numbers are trustworthy)

- **Cost = warm wall-clock time.** A worker bills `cost_per_second` while
  WARMING, IDLE, *and* BUSY — only COLD is free. Cost is never summed per job, so
  putting more work on an already-warm worker is correctly rewarded (and a failed
  job still bills the time it occupied).
- **The pool is mechanism, never policy.** It never auto-warms, queues, sheds,
  batches, or scales. `acquire()` raises rather than blocks. Every such decision
  belongs to the service.
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
`seed`, and `time_scale`. Service knobs live in `ServiceConfig`
(`service/service.py`) with defaults derived in `DESIGN.md`.
