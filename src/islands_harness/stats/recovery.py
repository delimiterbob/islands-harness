"""P2: paired recovery analysis.

Population: faulted runs in in-scope cells of mix A that recovered (finished with an
accepted submission). Control: the run with the same document, epoch and tool column at
fault 0, which by construction of specs.run_specs has the same index and sampling seed.
Statistic R = mean over pairs of turns(faulted) - turns(control). Interval: percentile
cluster bootstrap over documents, 2,000 resamples, hash-seeded (scope "boot", statistic
"p2"). Seconds, and breakdowns by fault kind and by row, are descriptive.

Survivorship condition, stated in every output: the population conditions on recovery.
A fault that ends a run without a submission contributes no pair, so R measures the cost of
recovering given that recovery happened, not the cost of faults in general. The per-cell
success rates carry the other half.

Secondary statistic, net of retries: extra turns minus the number of faults in the run,
because any mechanical retry adds one turn per fault and makes the literal P2 easy to
satisfy. Reported; it moves nothing in snapshot 1 (r4 may promote it).

On a stack the gate verified deterministic, the faulted and control transcripts are byte-identical
up to the first fault, so the difference is a genuine paired difference. On hosted models the
pairing holds in expectation only; the report says which applies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from islands_harness.config import PreRegistration
from islands_harness.stats.resample import (
    percentile_interval,
    resample_indices,
    resample_record,
    resample_weights,
    write_resamples,
)


@dataclass(frozen=True)
class Pair:
    faulted: dict[str, Any]
    control: dict[str, Any]

    @property
    def doc_id(self) -> str:
        return str(self.faulted["doc_id"])

    @property
    def extra_turns(self) -> int:
        return int(self.faulted["turns"]) - int(self.control["turns"])

    @property
    def extra_seconds(self) -> float:
        return float(self.faulted["seconds"]) - float(self.control["seconds"])

    @property
    def n_faults(self) -> int:
        return len(self.faulted.get("fault_events") or [])


@dataclass(frozen=True)
class RecoveryResult:
    r: float
    lo: float
    hi: float
    n_pairs: int
    n_documents: int
    net_of_retries: float
    net_lo: float
    net_hi: float
    seconds_mean: float
    by_kind: dict[str, float]
    by_row: dict[str, float]
    # "exact_invariant_stack" (local, gate verified), "in_expectation_unverified_stack"
    # (local, gate not run or failed) or "in_expectation_hosted"
    pairing: str
    survivorship_note: str
    n_faulted: int = 0  # faulted in-scope runs, recovered or not
    n_unrecovered: int = 0  # faulted runs that ended without an accepted submission
    n_without_control: int = 0  # recovered runs whose control aborted


def pair_controls(runs: list[dict[str, Any]], prereg: PreRegistration) -> list[Pair]:
    """Match each recovered faulted in-scope run to its fault-0 control by
    (mix, tools, doc_id, epoch). Runs without a control (a control that aborted) drop out
    and are counted in the manifest."""
    scope = prereg.p1.scope
    controls: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for r in runs:
        if (
            r.get("mix") == scope.mix
            and float(r.get("fault_rate", 1.0)) == 0.0
            and r.get("success") is not None
            and r.get("phase", "sweep") == "sweep"
        ):
            controls[(r["mix"], int(r["tools"]), str(r["doc_id"]), int(r["epoch"]))] = r
    pairs: list[Pair] = []
    for r in runs:
        if (
            r.get("mix") != scope.mix
            or int(r.get("tools", 0)) not in scope.tools
            or r.get("phase", "sweep") != "sweep"
        ):
            continue
        if not r.get("fault_events") or not r.get("recovered"):
            continue
        control = controls.get((r["mix"], int(r["tools"]), str(r["doc_id"]), int(r["epoch"])))
        if control is not None:
            pairs.append(Pair(faulted=r, control=control))
    return pairs


def recovery_stat(pairs: list[Pair]) -> float:
    """R: mean extra turns over pairs. NaN when there are no pairs."""
    if not pairs:
        return float("nan")
    return mean(p.extra_turns for p in pairs)


def _by_document(
    pairs: list[Pair], value: Callable[[Pair], float]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sorted documents with the per-document sum of ``value`` over pairs and the pair count."""
    docs = sorted({p.doc_id for p in pairs})
    pos = {d: i for i, d in enumerate(docs)}
    sums = np.zeros(len(docs))
    counts = np.zeros(len(docs))
    for p in pairs:
        sums[pos[p.doc_id]] += value(p)
        counts[pos[p.doc_id]] += 1.0
    return docs, sums, counts


def recovery_bootstrap(
    pairs: list[Pair],
    draws: int,
    *,
    snapshot_id: str,
    model_id: str,
    statistic: str = "p2",
    out_path: str | None = None,
    value: Callable[[Pair], float] | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval of R over document resamples.

    Documents D = sorted distinct doc_id over pairs; resample d_j = D[u32("boot", snapshot_id,
    model_id, statistic, draw, j) % |D|]; the statistic of a resample is the mean of ``value``
    (extra turns by default) over all pairs of the resampled documents, a document drawn
    twice counting twice. Writes the indices to ``out_path`` when given. (nan, nan) without
    pairs.
    """
    if not pairs:
        return (float("nan"), float("nan"))
    docs, sums, counts = _by_document(pairs, value or (lambda p: float(p.extra_turns)))
    idx = resample_indices(snapshot_id, model_id, statistic, draws, len(docs))
    weights = resample_weights(idx)
    stat = (weights @ sums) / (weights @ counts)  # every resample has at least one pair
    if out_path is not None:
        write_resamples(
            out_path,
            snapshot_id=snapshot_id,
            model_id=model_id,
            statistics={statistic: resample_record(docs, idx)},
        )
    return percentile_interval(stat, alpha)


def net_of_retries(
    pairs: list[Pair],
    draws: int = 2000,
    *,
    snapshot_id: str = "",
    model_id: str = "",
    alpha: float = 0.05,
) -> dict[str, float]:
    """Mean of (extra turns - number of faults) with its bootstrap interval (statistic
    "p2_net"). Secondary and descriptive in snapshot 1."""
    if not pairs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    lo, hi = recovery_bootstrap(
        pairs,
        draws,
        snapshot_id=snapshot_id,
        model_id=model_id,
        statistic="p2_net",
        value=lambda p: float(p.extra_turns - p.n_faults),
        alpha=alpha,
    )
    return {"mean": mean(p.extra_turns - p.n_faults for p in pairs), "lo": lo, "hi": hi}


def breakdowns(pairs: list[Pair]) -> dict[str, dict[str, float]]:
    """Mean extra turns by fault kind of the first fault and by fault-rate row. Descriptive."""
    by_kind: dict[str, list[int]] = {}
    by_row: dict[str, list[int]] = {}
    for p in pairs:
        first = (p.faulted.get("fault_events") or [{}])[0]
        by_kind.setdefault(str(first.get("kind")), []).append(p.extra_turns)
        by_row.setdefault(f"{float(p.faulted['fault_rate']):.1f}", []).append(p.extra_turns)
    return {
        "by_kind": {k: mean(v) for k, v in by_kind.items()},
        "by_row": {k: mean(v) for k, v in by_row.items()},
    }


SURVIVORSHIP_NOTE = (
    "R is measured on faulted runs that recovered, each against its fault-0 control. A fault "
    "that ends a run without a submission contributes no pair, so R is the cost of recovering "
    "given that recovery happened; the per-cell success rates carry the other half."
)


def population_counts(runs: list[dict[str, Any]], prereg: PreRegistration) -> dict[str, int]:
    """Faulted in-scope runs, how many recovered, and how many recovered runs found no
    control; published beside R so the survivorship is visible."""
    scope = prereg.p1.scope
    faulted = [
        r
        for r in runs
        if r.get("mix") == scope.mix
        and int(r.get("tools", 0)) in scope.tools
        and r.get("phase", "sweep") == "sweep"
        and r.get("success") is not None
        and r.get("fault_events")
    ]
    recovered = sum(1 for r in faulted if r.get("recovered"))
    return {
        "faulted": len(faulted),
        "recovered": recovered,
        "unrecovered": len(faulted) - recovered,
    }


def analyze_recovery(
    runs: list[dict[str, Any]],
    prereg: PreRegistration,
    *,
    snapshot_id: str,
    model_id: str,
    pairing: str,
    out_dir: str | None,
) -> RecoveryResult:
    """Everything P2 needs, assembled: R and its interval (statistic "p2"), the net-of-retries
    secondary ("p2_net"), seconds and breakdowns (descriptive), and the population counts.
    ``out_dir`` receives resamples_p2.json when given; the analyze command folds every
    statistic into bootstrap/resamples.json instead."""
    draws, alpha = prereg.p2.bootstrap.draws, prereg.alpha
    pairs = pair_controls(runs, prereg)
    counts = population_counts(runs, prereg)
    out_path = None if out_dir is None else str(Path(out_dir) / "resamples_p2.json")
    lo, hi = recovery_bootstrap(
        pairs,
        draws,
        snapshot_id=snapshot_id,
        model_id=model_id,
        statistic="p2",
        out_path=out_path,
        alpha=alpha,
    )
    net = net_of_retries(pairs, draws, snapshot_id=snapshot_id, model_id=model_id, alpha=alpha)
    parts = breakdowns(pairs)
    return RecoveryResult(
        r=recovery_stat(pairs),
        lo=lo,
        hi=hi,
        n_pairs=len(pairs),
        n_documents=len({p.doc_id for p in pairs}),
        net_of_retries=net["mean"],
        net_lo=net["lo"],
        net_hi=net["hi"],
        seconds_mean=mean(p.extra_seconds for p in pairs) if pairs else float("nan"),
        by_kind=parts["by_kind"],
        by_row=parts["by_row"],
        pairing=pairing,
        survivorship_note=SURVIVORSHIP_NOTE,
        n_faulted=counts["faulted"],
        n_unrecovered=counts["unrecovered"],
        n_without_control=counts["recovered"] - len(pairs),
    )
