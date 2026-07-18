# HILOS — Lead Platform Engineer Take-Home

**The Pipeline Runner**

Thanks for getting this far. This exercise is the closest thing we have to a day in the role: building the service layer that sits between our AI/ML pipelines and the people (and agents) who use them. We've designed it to be a real systems problem, not a trivia quiz.

---

## The challenge

HILOS runs AI pipelines that turn designer inputs into footwear assets — segment a photo, remove its background, generate multiple views, fit a 3D last. Each pipeline is a long-running job that may call a GPU model. Two very different consumers need to invoke these pipelines, watch their progress, and compose them into multi-step chains:

- the **product frontend**, which calls them directly for explicit user actions, and
- an **AI agent**, which reasons about what the user wants and composes pipelines into a chain.

GPUs are scarce and expensive, real model calls are slow and occasionally flaky, and demand is bursty. **Your job: build the service layer that runs these pipelines reliably and cost-effectively.**

You will *not* be building or training any models — we've stubbed those out (see below). We want to see how you think about scheduling, resource economics, reliability, and clean contracts: the production-engineering layer.

---

## Philosophy

- **We value judgment over completeness.** A sharp, well-reasoned partial system beats an attempt at everything. What you deliberately *skip or stub — and why —* is as informative to us as what you finish.
- There is no single correct architecture. Make decisions, justify them, and tell us what you'd revisit.

---

## What you'll get (starter kit)

A small repo containing mock "pipeline" functions so you can spend your effort on the hard part rather than on scaffolding:

- **Four pipeline functions**, each a pure-Python `async` function with a typed (Pydantic) input/output schema:
  - `segment(image) -> mask` — *CPU-bound, cheap, always available*
  - `remove_bg(image, mask) -> cutout` — *CPU-bound, cheap, always available*
  - `generate_multiview(cutout) -> views[8]` — *GPU-bound*
  - `fit_to_last(views) -> mesh` — *GPU-bound*
- Each function **sleeps for a configurable duration** (simulating real work) and **fails at a configurable rate** (simulating flaky models / transient errors).
- The two **GPU-bound** functions run on a **limited pool of GPU workers** with a real cost/latency model:
  - workers **cost money per second while they're warm** — whether or not they're doing work,
  - a worker spun up from cold pays a **cold-start penalty** (a configurable warm-up delay, e.g. ~30–60s) before its first job,
  - CPU-bound work is effectively free and instant to schedule.
- A **burst workload generator** that issues requests in spikes rather than uniformly, so you can see how your scheduling behaves — and what it costs — under realistic load.
- No real models, no actual GPU, no external services. Everything runs locally.

Everything is a knob: latencies, failure rates, pool size, cold-start duration, per-second cost, and burst shape.

---

## What to build

### Core — please attempt all of this

A service that:

1. **Accepts a request to run a single pipeline *or* a chain** (where the output of one feeds the next), via a typed API.
2. **Runs them asynchronously as jobs** — returns a job id immediately; the client can poll or stream status.
3. **Schedules GPU work cost-effectively under bursty load.** This is the heart of the exercise. GPU workers are limited, cost money while warm, and pay a cold-start penalty from idle; requests arrive in spikes. Respect the pool limit and apply backpressure, *and* make a deliberate policy for the **latency-vs-cost tradeoff** — how large a warm pool to hold, when (if ever) to scale to zero, whether to batch. Treat **cost as a first-class signal** alongside latency, and defend your policy in `DESIGN.md`.
4. **Handles failure deliberately** — retry-with-backoff on transient errors; *localized* reporting for a chain (which step failed, what succeeded); and idempotency so re-submitting the same job doesn't double-run it.
5. **Exposes a typed contract an *agent* could actually drive** — not just a frontend. We're protocol-agnostic (REST, an MCP-style tool schema, whatever you prefer); what we care about is the judgment: sensible *operation granularity* for an LLM (not a 1:1 dump of every CRUD endpoint), and how a job's progress, result, and **failures** are surfaced *so an agent can act on them*.
6. **Is observable** — structured logs, and metrics that include queue depth, job latency (p50/p95), GPU utilization, failure rate, **and running $ cost**. Ideally an end-to-end trace of a single chain.

> We intentionally don't prescribe your queue/worker technology, whether you use a datastore, how you model job state, your exact API shape, or how far to push observability. Pick, justify, and scope. That's the exercise.

### Go deep on one thing

Once the core works, pick **one** area and take it to genuine production depth — and tell us in `DESIGN.md` *why you chose it*. We're more interested in one thing done with real depth than in five things half-built, and what you choose to go deep on tells us what you think matters most.

Some directions (or choose your own):

- **Push the economics further** — request batching / coalescing for GPU calls, adaptive autoscaling, or a scheduling policy that responds to the observed burst pattern.
- **Type-safe chaining** — validate that step N's output is compatible with step N+1's input *at submit time*, so a bad chain fails fast instead of mid-run.
- **A clean resource-declaration abstraction** — tools *declare* their resource needs and the scheduler honors them, with no special-casing.
- **A real trace view** of a chain running step-by-step (timing, per-step status, and cost), CLI or UI.
- **A chaos / load harness** that drives the burst generator with injected failures and demonstrates how your system holds up — and what it costs — under stress.

---

## Deliverables

1. **A repository** with your running service, all code, and setup instructions.
2. **A `DESIGN.md`** (1–2 pages) covering:
   - the architecture and key tradeoffs, **including your GPU scheduling / cost policy and the reasoning behind it**,
   - **the one area you took to depth, and why you chose it**,
   - what you deliberately skipped, and what you'd tackle next,
   - **where you'd draw the line between a platform engineer and an ML engineer for this system** — e.g. who owns model serving, and how the handoff from experimentation to production should work,
   - a short **appendix (≤ half a page)** answering the interaction-loop design prompt below.
3. A **README** with: how to run it and any assumptions you made.

---

## Design appendix — capturing the interaction loop (sketch only)

The execution trace your system produces is, in effect, a record of *what happened*. Our real product generates a richer version: a designer makes a gesture, the agent proposes a chain, the user accepts / edits / rejects it, and an asset is (or isn't) kept downstream. We'd eventually like to use this interaction data for **LLM post-training / RL**, and don't yet have a crisp answer for capturing and storing it well.

In a **≤ half-page appendix** to your `DESIGN.md`, sketch how you'd extend this system to do that. We're interested in: **where** you'd capture (the points in the loop), the **event schema and how it's versioned** over time, how you'd **link a downstream outcome back to the decisions that produced it**, and how you'd handle **consent / PII / IP** for what is proprietary design work.

You do **not** need to design the reward function, labeling criteria, or training algorithm — that's an ML decision. Focus on the platform: capturing, structuring, storing, and versioning this data so it's available and trustworthy. (Telling us where you'd draw that platform/ML line is itself useful signal.) **No implementation expected.**

---

## A note on tools

**Use whatever you'd use on the job — including AI coding tools.** We're an AI-native team; pretending otherwise wouldn't tell us anything useful. You are the architect: we're evaluating your judgment and the result, and we'll ask you to explain every meaningful decision in the live walkthrough. Lean on AI to move fast, but own the design.

---

## How we'll evaluate

Roughly in priority order:

- **Scheduling & resource economics** — how well you manage the cost-vs-latency tradeoff for GPU work under bursty load, including cold starts.
- **Systems design & abstractions** — is the architecture sound, and are the abstractions clean and generalizing?
- **Reliability & failure handling** — retries, backpressure, idempotency, partial-chain recovery, behavior under load.
- **Observability** — can we see what the system is doing, what it costs, and debug a failed job?
- **Agent-facing contract & API design** — operation granularity and how jobs/errors are surfaced to a caller.
- **Judgment & communication** — your scoping, the depth of the area you chose, and your take on the platform/ML boundary.
- **Data & training-loop thinking** — the interaction-loop appendix.
- **Code quality & craft** — readable, idiomatic, tested where it matters, good DX.

---

We're genuinely looking forward to seeing how you approach this. Have fun with it.
