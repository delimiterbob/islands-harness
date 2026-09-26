"""Mechanical application of the r4 rules and the dated statement.

The verdict is code with fixture tests, not prose (D7, D8). Every condition is evaluated,
recorded with its value and its pass flag, and written to verdict.json beside the statement;
nothing here reads anything but the fit, the bootstrap, the cell table, the recovery result
and the frozen pre-registration (ARCHITECTURE.md Section 11).

P1 categories:

    survives          C1 curvature, C2 peak and C3 drop all hold
    restricted        b2's interval entirely above 0, or t*'s interval entirely outside
                      [2, 6], or the drop interval excludes the SESOI on the small side
    below_resolution  b2's interval includes 0 and the drop interval includes both 0 and
                      the SESOI (the equivalence reading of "intervals swallow every edge")
    not_estimable     the ladder ended there (no variation in scope)
    blocked           the fault-cap check failed (harness anomaly)

Residual branch. The named outcomes do not partition every case: for example the fitted
peak lies at or beyond six tools with an unbounded interval, or the curve declines across
the range without an interior peak, or C1 holds while the drop interval includes 0 but not
the SESOI. r4 names the complement (below_resolution.residual: below_resolution); such
cases carry the condition ``residual_branch`` so they stay visible, and the method note
reports how often the branch fires under simulated flat truths.

P2 categories: survives when the lower bootstrap bound of R is above 0; restricted when it
is not and the upper bound is below the SESOI in turns (an interval entirely below 0
included); below_resolution otherwise; not_estimable when no recovered faulted run has a control (r4 p2.no_pairs). The
secondary net-of-retries statistic moves nothing in snapshot 1.

Decision intervals follow the fit's rung: in the MLE rung C1 and the fault-cap check read
the cluster-robust Wald intervals; in the Firth rung they read the bootstrap percentile
intervals (r4 p1.ladder: firth_bootstrap). The t* interval is always the bootstrap one and
may have an infinite end when resamples without a peak exceed alpha / 2 (fit.py). In JSON,
infinite limits are written as the strings "inf" and "-inf" and NaN as null.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from islands_harness.config import PreRegistration
from islands_harness.stats.fit import (
    BootstrapResult,
    CellStat,
    FitResult,
    drop_column,
    fault_cap_check,
    newcombe,
    peak,
)
from islands_harness.stats.recovery import RecoveryResult


class Verdict(StrEnum):
    survives = "survives"
    restricted = "restricted"
    below_resolution = "below_resolution"
    not_estimable = "not_estimable"
    blocked = "blocked"

    def display(self) -> str:
        return self.value.replace("_", " ")


@dataclass(frozen=True)
class Condition:
    name: str
    value: Any
    passed: bool | None
    detail: str = ""


@dataclass(frozen=True)
class InstrumentResult:
    ok: bool
    anchor_p: float | None
    anchor_lo: float | None
    anchor_hi: float | None
    band: tuple[float, float]
    label: str  # "in_calibration" | "instrument_out_of_calibration" | "anchor_missing"


@dataclass(frozen=True)
class P1Verdict:
    verdict: Verdict
    conditions: list[Condition]
    peak: float | None
    peak_interval: tuple[float, float] | None
    drop_points: float | None
    drop_interval: tuple[float, float] | None
    estimator: str


@dataclass(frozen=True)
class P2Verdict:
    verdict: Verdict
    conditions: list[Condition]
    r: float
    interval: tuple[float, float]
    survivorship_note: str = ""


def _find(cells: list[CellStat], tools: int, fault: float) -> CellStat | None:
    for c in cells:
        if c.tools == tools and abs(c.fault - fault) < 1e-9:
            return c
    return None


def instrument_check(
    cells: list[CellStat],
    prereg: PreRegistration,
    *,
    anchor_tools: int = 2,
    anchor_fault: float = 0.0,
) -> InstrumentResult:
    """The anchor cell's rate must be inside the calibration band by no more than its Wilson
    half-width; otherwise the snapshot is instrument_out_of_calibration."""
    lo_b, hi_b = prereg.success.band
    anchor = _find(cells, anchor_tools, anchor_fault)
    if anchor is None:
        return InstrumentResult(False, None, None, None, (lo_b, hi_b), "anchor_missing")
    half = (anchor.hi - anchor.lo) / 2.0
    ok = (lo_b - half) <= anchor.p <= (hi_b + half)
    return InstrumentResult(
        ok,
        anchor.p,
        anchor.lo,
        anchor.hi,
        (lo_b, hi_b),
        "in_calibration" if ok else "instrument_out_of_calibration",
    )


P1_PRECEDENCE = ["not_estimable", "blocked", "survives", "restricted", "below_resolution"]


def verdict_p1(
    fit: FitResult, boot: BootstrapResult | None, cells: list[CellStat], prereg: PreRegistration
) -> P1Verdict:
    """The first category in P1_PRECEDENCE whose rule holds. r4 states the order
    (p1.precedence); a pre-registration naming another order is refused."""
    rules = prereg.p1
    if list(rules.precedence) != P1_PRECEDENCE:
        raise ValueError(f"r4 p1.precedence must be {P1_PRECEDENCE}, the implemented order")
    conds: list[Condition] = []

    if fit.status == "not_estimable":
        conds.append(Condition("estimable", False, False, fit.reason or "no variation in scope"))
        return P1Verdict(Verdict.not_estimable, conds, None, None, None, None, fit.status)

    intervals = fit.wald_intervals if fit.status == "mle" else (boot.coef_intervals if boot else {})
    cap_ok = fault_cap_check(fit, intervals)
    conds.append(
        Condition("fault_cap", fit.coefficients.get("f"), cap_ok, "b3 <= 0 or interval includes 0")
    )
    if not cap_ok:
        return P1Verdict(Verdict.blocked, conds, None, None, None, None, fit.status)

    b1, b2 = fit.coefficients["tc"], fit.coefficients["tc2"]
    b2_lo, b2_hi = intervals.get("tc2", (float("-inf"), float("inf")))
    c1 = b2 < 0 and b2_hi < 0
    conds.append(
        Condition(
            "C1_curvature",
            {"b2": b2, "interval": [b2_lo, b2_hi]},
            c1,
            "b2 < 0 with interval excluding 0",
        )
    )

    t_star = peak(b1, b2)
    peak_iv = boot.peak_interval if boot else None
    lo_r, hi_r = rules.peak.range
    c2 = peak_iv is not None and lo_r <= peak_iv[0] and peak_iv[1] <= hi_r
    conds.append(
        Condition(
            "C2_peak",
            {"peak": t_star, "interval": list(peak_iv) if peak_iv else None},
            c2,
            f"bootstrap interval inside [{lo_r:g}, {hi_r:g}]",
        )
    )

    drop_pts: float | None = None
    drop_iv: tuple[float, float] | None = None
    c3 = False
    col = drop_column(t_star, prereg)
    a, b = _find(cells, col, rules.drop.row), _find(cells, int(hi_r), rules.drop.row)
    if col != int(hi_r) and a is not None and b is not None and a.n > 0 and b.n > 0:
        d_lo, d_hi = newcombe(a.k, a.n, b.k, b.n)
        drop_pts = 100.0 * (a.p - b.p)
        drop_iv = (100.0 * d_lo, 100.0 * d_hi)
        c3 = t_star is not None and drop_iv[0] > 0
    conds.append(
        Condition(
            "C3_drop",
            {"column": col, "points": drop_pts, "interval": list(drop_iv) if drop_iv else None},
            c3,
            "Newcombe interval for p(column) - p(6) excludes 0 in the zero row; the column is "
            "round(t*) clipped to [2, 6], or 2 without a peak",
        )
    )

    sesoi = rules.drop.sesoi_points
    if c1 and c2 and c3:
        verdict = Verdict.survives
    elif (
        b2_lo > 0
        or (peak_iv is not None and (peak_iv[1] < lo_r or peak_iv[0] > hi_r))
        or (drop_iv is not None and drop_iv[1] < sesoi)
    ):
        verdict = Verdict.restricted
        conds.append(
            Condition(
                "restricted_reason",
                None,
                None,
                "b2 interval above 0, or peak interval outside range, or drop interval below SESOI",
            )
        )
    elif (
        (b2_lo <= 0 <= b2_hi)
        and drop_iv is not None
        and drop_iv[0] <= 0 <= drop_iv[1]
        and drop_iv[0] <= sesoi <= drop_iv[1]
    ):
        verdict = Verdict.below_resolution
        conds.append(
            Condition(
                "sesoi_equivalence",
                {"sesoi_points": sesoi},
                True,
                "b2 interval includes 0; drop interval includes 0 and the SESOI",
            )
        )
    else:
        verdict = Verdict.below_resolution
        conds.append(
            Condition(
                "residual_branch",
                None,
                None,
                "no named branch matched; below resolution by r4 below_resolution.residual",
            )
        )
    return P1Verdict(verdict, conds, t_star, peak_iv, drop_pts, drop_iv, fit.status)


def verdict_p2(recovery: RecoveryResult, prereg: PreRegistration) -> P2Verdict:
    sesoi = prereg.p2.sesoi_turns
    if recovery.n_pairs == 0:
        conds = [Condition("n_pairs", 0, False, "no recovered faulted run with a control")]
        return P2Verdict(
            Verdict.not_estimable,
            conds,
            recovery.r,
            (recovery.lo, recovery.hi),
            recovery.survivorship_note,
        )
    conds = [
        Condition("lower_bound_above_zero", recovery.lo, recovery.lo > 0, "survives when true"),
        Condition(
            "interval_includes_zero",
            [recovery.lo, recovery.hi],
            recovery.lo <= 0 <= recovery.hi,
            "",
        ),
        Condition("upper_below_sesoi", recovery.hi, recovery.hi < sesoi, f"SESOI {sesoi:g} turns"),
        Condition("n_pairs", recovery.n_pairs, None, "survivorship: recovered runs only"),
    ]
    if recovery.lo > 0:
        v = Verdict.survives
    elif recovery.hi < sesoi:  # includes an interval entirely below 0 (r4 p2.sesoi_turns)
        v = Verdict.restricted
    else:
        v = Verdict.below_resolution
    return P2Verdict(v, conds, recovery.r, (recovery.lo, recovery.hi), recovery.survivorship_note)


def _fmt_iv(iv: tuple[float, float] | None, nd: int = 1) -> str:
    if iv is None:
        return "[n/a]"
    return f"[{iv[0]:.{nd}f}, {iv[1]:.{nd}f}]"


def jsonable(obj: Any) -> Any:
    """A copy safe for strict JSON: tuples become lists, NaN becomes None and infinities
    become the strings "inf" and "-inf"."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        return jsonable(obj.item())
    return obj


def statement(date: str, snapshot: str, model: str, mix: str, p1: P1Verdict, p2: P2Verdict) -> str:
    """One line in the fixed pipe-separated format, for example:

    2026-11-15 | s1 | gpt-oss-20b-local | mix A | P1: survives (peak 3.4 tools [2.9, 4.1]; drop 41 points [27, 53]) | P2: below resolution (+0.4 turns [-0.6, 1.3])
    """
    if p1.verdict in (Verdict.not_estimable, Verdict.blocked):
        p1_part = f"P1: {p1.verdict.display()}"
    else:
        peak_txt = (
            f"peak {p1.peak:.1f} tools {_fmt_iv(p1.peak_interval)}"
            if p1.peak is not None
            else "peak undefined"
        )
        drop_txt = (
            f"drop {p1.drop_points:.0f} points {_fmt_iv(p1.drop_interval, 0)}"
            if p1.drop_points is not None
            else "drop undefined"
        )
        p1_part = f"P1: {p1.verdict.display()} ({peak_txt}; {drop_txt})"
    if p2.verdict == Verdict.not_estimable:
        p2_part = "P2: not estimable"
    else:
        p2_part = f"P2: {p2.verdict.display()} ({p2.r:+.1f} turns {_fmt_iv(p2.interval)})"
    return f"{date} | {snapshot} | {model} | mix {mix} | {p1_part} | {p2_part}"


@dataclass
class VerdictFile:
    schema_version: int
    date: str
    snapshot: str
    model: str
    mix: str
    instrument: dict[str, Any]
    p1: dict[str, Any]
    p2: dict[str, Any]
    statement: str
    prereg_hash: str
    notes: list[str] = field(default_factory=list)
    propositions_move: bool = True  # False when the instrument check failed


def write_verdict_json(
    path: Path,
    *,
    date: str,
    snapshot: str,
    model: str,
    mix: str,
    instrument: InstrumentResult,
    p1: P1Verdict,
    p2: P2Verdict,
    prereg_hash: str,
) -> str:
    """Write verdict.json and return the statement line. If the instrument check failed, both
    propositions carry ``moves: false`` (and ``propositions_move`` is false), so a reader of
    the file cannot publish a category that did not move, and the statement says so."""
    if not instrument.ok:
        line = f"{date} | {snapshot} | {model} | mix {mix} | {instrument.label}: neither proposition moves"
    else:
        line = statement(date, snapshot, model, mix, p1, p2)
    payload = VerdictFile(
        schema_version=1,
        date=date,
        snapshot=snapshot,
        model=model,
        mix=mix,
        instrument=asdict(instrument),
        p1={**asdict(p1), "verdict": p1.verdict.value, "moves": instrument.ok},
        p2={**asdict(p2), "verdict": p2.verdict.value, "moves": instrument.ok},
        statement=line,
        prereg_hash=prereg_hash,
        propositions_move=instrument.ok,
    )
    Path(path).write_text(
        json.dumps(jsonable(asdict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return line
