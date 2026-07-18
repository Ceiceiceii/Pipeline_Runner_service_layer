"""Deterministic primitives for reproducible simulation.

Every random decision in the kit — whether a step fails, how much latency jitter
it gets, what the burst schedule looks like — derives from a single integer seed
plus a stable hash of contextual keys (step name, input id, attempt number).
This makes each outcome a *pure function of its inputs*, so concurrent asyncio
interleaving can never change a run's results: "step X on input Y, attempt K"
fails (or not) identically every time.

We deliberately avoid Python's builtin :func:`hash`, which is salted per process
(``PYTHONHASHSEED``). A take-home graded on Linux must reproduce a candidate's
Windows run byte-for-byte, so all hashing goes through :mod:`hashlib`.
"""

from __future__ import annotations

import hashlib
import random

_DIGEST_SIZE = 8  # 64 bits is plenty for ids and seeds


def _canonical(parts: tuple[object, ...]) -> bytes:
    """Encode ``parts`` to a stable byte string (NUL-separated UTF-8)."""
    return b"\x00".join(str(part).encode("utf-8") for part in parts)


def stable_hash(*parts: object) -> int:
    """Return a stable, process-independent 64-bit integer hash of ``parts``."""
    digest = hashlib.blake2b(_canonical(parts), digest_size=_DIGEST_SIZE).digest()
    return int.from_bytes(digest, "big")


def rng_for(seed: int, *key: object) -> random.Random:
    """Return a :class:`random.Random` seeded deterministically from ``seed`` and ``key``.

    Each distinct ``key`` yields an independent stream, so drawing from one does
    not perturb another regardless of the order in which coroutines run.
    """
    return random.Random(stable_hash(seed, *key))


def content_id(*parts: object) -> str:
    """Return a short, stable, content-addressed id derived from ``parts``.

    The same inputs always produce the same id, which is what makes pipeline
    outputs content-addressable: re-running ``segment`` on the same image yields
    a mask with an identical id, giving idempotency a natural hook.
    """
    return hashlib.blake2b(_canonical(parts), digest_size=_DIGEST_SIZE).hexdigest()
