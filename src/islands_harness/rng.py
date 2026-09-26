"""Stateless, hash-derived randomness. Every random quantity in the harness comes from here.

There is no module-level state, no ``random.Random`` object and no generator to advance: a
draw is a pure function of a scope string and the parts that identify it, so the same draw
gives the same value on every platform, in any order, at any concurrency, and a reviewer can
recompute any single draw from its indices (ARCHITECTURE.md D5).

    value = f(sha256(canonical_json([scope, *parts])))

Canonical encoding: ``json.dumps([scope, *parts], separators=(",", ":"), sort_keys=True,
ensure_ascii=True, allow_nan=False)`` encoded as UTF-8. Parts must be JSON-native scalars or
containers of them (str, int, bool, None, lists, dicts). Floats are allowed but discouraged in
scopes because their textual form is what is hashed; the harness passes rates as ints
(percent) or fixed strings where they must enter a scope.

Scopes in use (grep for ``unit("`` / ``seed64("`` to find each):

    sample   per-run sampling seed:            ("sample", root_seed, model_id, run_index)
    fault    fault decision u1:                ("fault", root_seed, model_id, mix, doc_id, epoch, call_index)
    kind     fault kind u2:                    ("kind", same indices as fault)
    garble   garble positions and truncation:  ("garble", same indices as fault, k)
    perm     document permutation:             ("perm", root_seed, model_id, i)
    boot     bootstrap resample indices:       ("boot", snapshot_id, model_id, statistic, draw, j)
    filler   determinism-gate filler requests: ("filler", root_seed, model_id, request_index)
    search   canned search-result selection:   ("search", doc_id, query)
    email    fake email message ids:           ("email", doc_id, to, subject)
    dataset  synthetic invoice generation:     ("dataset", seed, difficulty, document_number, field)
    varied   noise-floor varied-seed offset:   ("varied", rerun_index) appended to sample parts

The fault scope deliberately omits tool count and fault rate so faults nest across rates
(ARCHITECTURE.md Section 7).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

_TWO_53 = float(1 << 53)


def _digest(scope: str, parts: tuple[Any, ...]) -> bytes:
    payload = json.dumps(
        [scope, *parts], separators=(",", ":"), sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def unit(scope: str, *parts: Any) -> float:
    """A float in [0, 1) from the top 53 bits of sha256 over the canonical encoding.

    53 bits is the mantissa width of a double, so every representable value in [0, 1) with
    that resolution is reachable and the mapping is exact (no rounding, no bias).
    """
    top = int.from_bytes(_digest(scope, parts)[:8], "big") >> 11
    return top / _TWO_53


def u32(scope: str, *parts: Any) -> int:
    """An unsigned 32-bit integer: the first four digest bytes, big-endian."""
    return int.from_bytes(_digest(scope, parts)[:4], "big")


def seed64(scope: str, *parts: Any) -> int:
    """A non-negative integer that fits a signed 64-bit field on every wire format.

    The first eight digest bytes, big-endian, shifted right by one (63 bits), so it is valid
    as an OpenAI-compatible ``seed`` (int64) and as a torch seed.
    """
    return int.from_bytes(_digest(scope, parts)[:8], "big") >> 1


def choice_weighted(u: float, items: Sequence[Any], weights: Sequence[float]) -> Any:
    """Map a uniform ``u`` in [0, 1) through the cumulative weights to one of ``items``.

    Pure: the same ``u`` always selects the same item. Weights must be non-negative with a
    positive sum. The last item absorbs floating-point slack at the top of the range.
    """
    if len(items) != len(weights) or not items:
        raise ValueError("items and weights must be non-empty and the same length")
    total = float(sum(weights))
    if total <= 0 or any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative with a positive sum")
    if not 0.0 <= u < 1.0:
        raise ValueError("u must be in [0, 1)")
    threshold = u * total
    acc = 0.0
    for item, w in zip(items, weights, strict=True):
        acc += float(w)
        if threshold < acc:
            return item
    return items[-1]
