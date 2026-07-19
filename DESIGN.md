# DESIGN — Pipeline Runner service layer

## Architecture

A single-process asyncio service in `service/`. The kit is untouched. There is one `KitSettings`, one `Clock`, and one `GpuPool`, injected everywhere, so a time-compressed demo and a virtual-clock test bill the same dollars the kit meters.

```text
frontend ──► FastAPI REST (+SSE)  ┐
agent ─────► tool layer (4 ops)   ┴─► JobService ─► ChainRunner ─┬─ CPU steps inline
                                        │                        └─ GPU steps ─► GpuScheduler
                                   JobStore (Protocol,               bounded FIFO + policy-sized
                                   in-memory impl)                   worker-runner tasks ─► GpuPool
```

The job model is one structure, `Job → StepRun → AttemptSpan`, recording every attempt with timestamps and worker id. The status payload, the trace, and the metrics are all views over it. Errors carry `{step_name, attempt, kind, retryable}` so a caller can act on them without parsing strings. Chains are validated against the `STEPS` registry at submit time; a bad chain gets a 422 naming the incompatible pair instead of failing minutes later mid-run. Idempotency uses the client's key, or falls back to `content_id(chain, image)` with the kit's own hashing, and a duplicate submit returns the existing job. Step outputs are memoized by `(step, input_ids)`, which the content-addressed ids make safe, so a resubmitted or overlapping chain skips straight to the first missing step. Steps route on a declared `ResourceClass` in `service/resources.py`. An unclassified step is an import-time error, because the silent alternative is a GPU step running inline, unmetered.

## Deep dive: scheduling and resource economics

I went deep here for two reasons. It is the first thing Hilos says it evaluates, and the kit makes it measurable: deterministic load plus a billed clock means every scheduling claim below has a dollar figure attached, reproducible from a clean clone.

**The numbers that drive everything** (verified against the kit, defaults): 160 requests over 120s in 3 spikes; E[GPU-work per arrival] = 0.6·20s + 0.25·8s = **14 GPU-s**; total ≈ 2,490 GPU-s (with retries) vs 4 GPU-s/s of pool capacity ⇒ **~5× oversubscribed** while arrivals last; the queue cannot drain between bursts. Cold start E≈45s is billed, so keep-warm break-even is 45s > the 30s quiet gap.

**Mechanism: long-lived worker-runners.** The policy sets a target warm count and the scheduler maintains that many runner tasks. Each acquires a worker once, warms it once, then drains the queue until an idle TTL expires. Coalescing needs no extra code this way: one cold start amortizes across every job the runner drains, where per-job leasing would pay ~45s of billed warm-up for each ~20s job. Transient retries run on the held worker with capped backoff (1s up to 15s). I considered requeueing retries instead and decided against it. The worker bills while idle anyway, so backoff-in-place costs the same money, but requeueing would send the unlucky 10% of jobs to the back of a multi-minute line.

**Scale-up: drain-time targeting.** `target = ceil(queue_depth × EWMA(service_s) / 60s)`, with the EWMA seeded from the registry's nominal timings. I looked at three alternatives before settling on this. Utilization thresholds saturate at 100% the moment a burst lands and then say nothing about how much excess demand exists. Erlang-C assumes a stationary arrival process, which this workload deliberately is not. Schedule-aware prediction would fit the shipped burst pattern, but the burst shape is a config knob, and a policy that memorizes it breaks the moment someone turns the knob. Queue depth leads; everything else lags. One knob, and it means something an operator can argue about: the backlog should clear within a minute.

**Scale-down: a ski-rental TTL.** A runner cools after 60 seconds idle (`min_warm=0`), the expected cold start plus a margin. Keep-warm versus cool is the classic ski-rental problem, and pricing the TTL at break-even is the standard answer: no forecasting, worst case bounded at twice the offline optimum. Because the break-even (45s) exceeds the quiet gap (30s), the pool rides mid-run lulls warm and reaches zero only after the final drain. Nothing in the code knows the burst schedule. Change it, and the same arithmetic adapts.

**Backpressure: demand-based admission, owned by the scheduler.** The bound counts admitted-but-unfinished GPU items, reserved for the whole chain at submit: `max_backlog = capacity × 300s promise / 10 GPU-s per item = 120 items`, about 60 full chains. Counting admitted demand rather than visible queue depth closes two holes a plain queue bound has. A burst of jobs still in their ~1.3s CPU prelude contributes nothing to queue depth, so a depth check admits all of it; a reservation does not. And cancelled work frees its slot immediately instead of sitting in the queue as a tombstone. A rejected submit gets a typed 503 with a computed `retry_after`. Mid-chain steps of already-admitted jobs are never rejected, since the GPU seconds spent on them are sunk. Under default load the bound has to engage in bursts two and three (about 47 GPU-bearing jobs arrive per cycle against ~10 drained), so the ~50 rejections in the table below are the wait promise being kept. CPU-only chains skip admission entirely. I would rather shed loudly at the door than let an unbounded queue turn every wait estimate into a lie.

**Evidence.** `python -m service.harness`, seed 0, identical replayed schedule per row. The `naive` row is the kit used raw, demo-style: per-job lease, own cold start, no queue, no retries. That is what exists before any service layer, so it is the honest baseline. Always-warm is cut off at its last job, which biases the comparison in its favor.

```text
scenario=default seed=0
policy            total $  $/success    p50 s    p95 s  cold   util   rej    ok  fail  makespan
-----------------------------------------------------------------------------------------------
naive              0.1804    0.00531      1.4     98.7    11   0.17   126    34     0     204.0
adaptive           0.5843    0.00546    222.2    359.2     4   0.78    53   107     0     458.0
adaptive-warm1     0.5192    0.00485    224.2    356.5     4   0.88    53   107     0     457.0
always4            0.5664    0.00501    214.9    355.3     4   0.88    47   113     0     472.0
eager0             0.5183    0.00484    222.2    359.2     4   0.88    53   107     0     458.0

scenario=sparse seed=0  (lone jobs ~50s apart: the idle-dominated regime)
policy            total $  $/success    p50 s    p95 s  cold   util   rej    ok  fail  makespan
-----------------------------------------------------------------------------------------------
naive              0.2848    0.03165     92.3    122.0    18   0.18     0     9     2     688.0
adaptive           0.1644    0.01495     33.0     64.0     2   0.35     0    11     0     640.0
adaptive-warm1     0.1920    0.01745     26.0     53.4     1   0.30     0    11     0     640.0
always4            0.7500    0.06818     21.0     38.4     4   0.08     0    11     0     625.0
eager0             0.1314    0.01195     51.0     90.0     5   0.44     0    11     0     675.0
```

One reading note on the naive row: its default-scenario p50 of 1.4s is survivor bias, not speed. 126 of 160 requests died at the full pool and never enter the percentile, and most survivors are CPU-only cutout chains. Read `$/success` next to `ok` instead of raw totals; a scheduler that refuses work always looks cheap and fast.

The two regimes tell different stories, and I think showing both matters more than showing the flattering one. Under saturation the policies converge to within about 15% of each other, because there is no idle time for a clever policy to manage. `always4` even wins on latency by pre-warming, and rejects the least, since faster draining frees admission slots sooner. Adaptive's small cost gap there is its TTL tail plus later warm-up, the price of not knowing the future. The sparse regime is where the policies separate: `always4` burns 4.5× the money on idle rent, `eager0` is cheapest but adds a ~45s cold start to nearly every job's latency, and adaptive lands near `always4`'s latency at under a quarter of its cost. Same policy, same knobs, both regimes. The `adaptive-warm1` row prices the interactive floor for anyone who wants it: p50 drops 7 seconds for about three cents. A chaos run (30% GPU failure) shows what reliability costs here: adaptive absorbs it for +16% fleet cost and zero failed jobs, where the baseline kills two chains outright. Rankings hold across seeds 0–3.

Every claim in this section regenerates in seconds on the virtual clock: `python -m service.harness` for the tables above, `--scenario chaos` for the reliability numbers, `--seed 1` through `--seed 3` for ranking stability.

## Hardening pass (post-build self-review)

After the initial build passed its 50 tests and produced these tables, I ran a multi-angle review against my own code. It found 11 real correctness bugs, each now fixed and pinned by a dedicated regression test (61 tests total). Three mattered most. Cancellation mid-step could leave a job non-terminal forever, which broke shutdown. The admission bound originally counted queued items, so a burst still in its CPU prelude could slip past it entirely; the reserved-demand design above is the fix. And the harness itself had a billing bug that overstated the naive baseline's cost. That last one deserves a plain sentence: the bug flattered my own design, and fixing it made the published comparison less favorable to the service. The tables above are the corrected ones.

The same pass collapsed six duplicated definitions (error taxonomy, the clock-advance loop, hashing, the economics projections) into single sources, and turned three fail-open seams into loud typed errors: unclassified steps, per-process pools under `--workers N`, and submit before start.

What this bought, and what it cost. The core invariant, that a job task never ends non-terminal, now holds on every path. That is what makes the O(1) all-terminal check safe, and it lets cancel free GPU dollars and admission slots immediately. The 300s wait promise is enforceable under the exact burst shape shipped. The price is explicit state machinery: a reservation counter and per-job map that must balance on every completion path (releases are tied to future resolution rather than happy-path code, because a leak here would slowly choke admission), a sixth step state, and a runner-slot counter whose correctness depends on decrement-in-`finally` discipline. Memoized GPU steps briefly over-reserve, so admission is slightly conservative. Fail-closed is also stricter to work with: new steps must declare a resource class, the API runs factory-only, and tests must call `start()`. The regression suite exists so the subtle parts, like re-raising inside a `CancelledError` handler, don't get "simplified" back into bugs later.

## Deliberate skips, and what I'd do next

Durable storage (the `JobStore` Protocol is the seam; the first production change is Postgres plus a broker for the queue). Auth, multi-tenancy, and per-client fairness (today one chatty client can fill the backlog). Priority classes and deadlines (the mechanism slot is a priority-queue swap; I'd evaluate it with the harness before building it). Mid-attempt preemption (the kit offers none, so cancel is best-effort and the tool schema says so). Per-attempt timeouts. Streaming percentile sketches (a bounded reservoir is exact at this scale).

## Platform / ML boundary

Platform owns the serving substrate: scheduling, cost, retries and backpressure, idempotency, observability, and the data-capture path. ML owns step semantics: model quality, input/output schemas, and deciding which failures are retryable, because that is a model-knowledge question, not an infrastructure one. The handoff is the step registry entry. An ML engineer promotes a step to production by declaring its typed I/O, nominal timing, and resource class, the same contract the scheduler, the validator, and the agent tools already consume, so no platform code changes per model. Promotion from experiment to production means registering a step behind a version, and the platform's side of the deal is that the step inherits scheduling, retries, tracing, and cost attribution for free.

## Appendix: capturing the interaction loop

Capture at five points: the designer's gesture, the agent's proposed chain with its stated rationale, the accept/edit/reject decision (with the edit diff), the execution trace this system already produces, and the downstream outcome (asset kept, discarded, or exported).

Events go into an append-only envelope, `{event_id, schema_version, tenant, session, actor, ts, type, payload}`, with payloads versioned per event type. Migrations are forward-only: history is never rewritten, new readers handle `v+1`.

Linkage is a chain of causal ids: `gesture_id → proposal_id → job_id → artifact_id → outcome`. Artifact ids are content-addressed (already the kit's idiom), which makes the outcome join cheap and dedupe-safe. A training example is a walk along this chain.

For consent, PII, and IP: consent is a per-tenant flag enforced at export rather than capture, since you cannot retroactively capture what you skipped. PII scrubbing happens at ingest. Storage is partitioned by tenant, so a deletion request is a partition drop. Proprietary design geometry never enters the event stream, only opaque content ids; the assets themselves live in tenant-scoped storage under their own ACL. Reward design, labeling criteria, and the training algorithm are ML decisions. The platform's promise is narrower and harder: the record is complete, versioned, and trustworthy.
