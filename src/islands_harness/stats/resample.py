"""Document resampling shared by every bootstrap in the harness.

One draw of a cluster bootstrap picks D documents with replacement from the sorted list of D
documents. Index j of draw b is

    u32("boot", snapshot_id, model_id, statistic, b, j) % D

so any single index can be recomputed from its coordinates by an independent implementation
(rng.py, ARCHITECTURE.md Section 9). The indices are written to ``bootstrap/resamples.json``.

The modulo introduces a bias of at most D / 2^32 per index (under 3e-8 for D = 100), far
below anything a percentile interval can resolve; it is kept because it makes the rule a
one-liner that any language reproduces exactly.

Percentile intervals are symmetric order statistics: with B draws sorted and
k = floor(B alpha / 2), the limits are the (k+1)-th smallest and the (k+1)-th largest draw.
For 2,000 draws at alpha = 0.05 that is the 51st draw from each end, so an end of the
interval is infinite exactly when more than k draws (more than 2.5 percent) are infinite on
that side. No interpolation takes place, so infinite draws (a resample without a peak, see
fit.py) order correctly instead of producing NaN.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from islands_harness.rng import u32


@lru_cache(maxsize=64)
def _indices(snapshot_id: str, model_id: str, statistic: str, draws: int, n: int) -> np.ndarray:
    out = np.empty((draws, n), dtype=np.int64)
    for b in range(draws):
        for j in range(n):
            out[b, j] = u32("boot", snapshot_id, model_id, statistic, b, j) % n
    out.setflags(write=False)
    return out


def resample_indices(
    snapshot_id: str, model_id: str, statistic: str, draws: int, n: int
) -> np.ndarray:
    """A read-only (draws, n) array of document positions. Cached: the coverage experiment
    reuses one index table across simulated snapshots, which is valid because the snapshots'
    data are independent of the indices."""
    if draws < 1 or n < 1:
        raise ValueError(f"need draws >= 1 and n >= 1, got {draws}, {n}")
    return _indices(snapshot_id, model_id, statistic, draws, n)


def resample_weights(indices: np.ndarray) -> np.ndarray:
    """(draws, n) multiplicities: how many times each document appears in each draw."""
    draws, n = indices.shape
    flat = (indices + n * np.arange(draws)[:, None]).ravel()
    return np.bincount(flat, minlength=draws * n).reshape(draws, n).astype(float)


def percentile_interval(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Symmetric order-statistic interval (module docstring); values may contain +inf
    and -inf, never NaN."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("no draws")
    if np.isnan(v).any():
        raise ValueError("NaN in bootstrap draws; map undefined draws before calling")
    v = np.sort(v)
    k = min(int(math.floor(v.size * alpha / 2 + 1e-9)), (v.size - 1) // 2)
    return (float(v[k]), float(v[v.size - 1 - k]))


def write_resamples(
    path: Path | str,
    *,
    snapshot_id: str,
    model_id: str,
    statistics: dict[str, dict[str, Any]],
) -> None:
    """Write ``{"snapshot_id", "model_id", "statistics": {name: {"documents", "draws",
    "indices"}}}`` with sorted keys and LF line endings, so two runs produce identical bytes.
    """
    payload = {"snapshot_id": snapshot_id, "model_id": model_id, "statistics": statistics}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def resample_record(documents: list[str], indices: np.ndarray) -> dict[str, Any]:
    """The per-statistic entry of resamples.json."""
    return {
        "documents": list(documents),
        "draws": int(indices.shape[0]),
        "indices": indices.tolist(),
    }
