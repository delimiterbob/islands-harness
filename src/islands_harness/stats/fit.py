"""Intervals, the pre-registered P1 fit ladder, bootstraps, the fault-cap check, the
fault-axis shape fit and the sensitivity analysis.

Every function here is a pure function of runs.jsonl rows (as dicts) and the frozen
pre-registration; alpha = 0.05 unless the prereg says otherwise; every random procedure
is hash-seeded through rng and its resample indices are written out. Formulas
(ARCHITECTURE.md Section 9):

Wilson score interval, k successes in n runs, p = k/n, z = 1.959964 at 95 percent:

    centre     = (p + z^2 / 2n) / (1 + z^2 / n)
    half-width = z * sqrt(p (1 - p) / n + z^2 / 4n^2) / (1 + z^2 / n)

    from statsmodels.stats.proportion.proportion_confint(method="wilson"); about 9.7 points
    at n = 100 near p = 0.5.

Newcombe hybrid score interval (1998, method 10) for d = p1 - p2, with (l1, u1) and (l2, u2)
the Wilson limits of each proportion:

    lower = d - sqrt((p1 - l1)^2 + (u2 - p2)^2)
    upper = d + sqrt((u1 - p1)^2 + (p2 - l2)^2)

Primary fit for P1 (one model, mix A, in-scope runs, aborted_transport excluded):

    logit P(success) = b0 + b1 tc + b2 tc2 + b3 f,   tc = tools - 4, tc2 = tc^2, f = fault rate
    statsmodels GLM(family=Binomial()) at run level, cov_type="cluster", groups = document id
    peak t* = 4 - b1 / (2 b2), defined only when b2 < 0

Fit ladder: MLE with cluster-robust Wald intervals; if not converged or any |b_j| > 15,
Firth's penalized likelihood (firthmodels.FirthLogisticRegression) with cluster bootstrap
intervals for everything; if Firth fails (no variation in scope), status ``not_estimable``.
A saturated column alone never triggers the ladder.

Cluster bootstrap: 2,000 resamples of documents with replacement, refit each time; the
intervals for t* and for the drop are percentile intervals; resamples where b2 >= 0 are
counted (their fraction reported), not dropped. Indices come from
``rng.u32("boot", snapshot_id, model_id, statistic, draw, j) % D`` and are written to
bootstrap/resamples.json so an independent implementation reproduces the intervals exactly.

Fault-cap consistency: b3 <= 0, or its interval includes 0; otherwise the verdict is
blocked and a harness anomaly reported.

Fault-axis shape (descriptive), per in-scope column and pooled, six cell rates fitted by
maximum binomial likelihood:

    LIN   p(f) = clip(a + b f, eps, 1 - eps)
    LOG3  p(f) = a / (1 + exp(s (f - f0)))
    AIC = 2k - 2 logL (k = 2 and 3); report AIC_LIN - AIC_LOG3, f0, s;
    cliff if >= 2, slope if <= -2, indistinguishable otherwise;
    scipy.optimize.minimize (bounded Nelder-Mead) from every start of a fixed grid, with the
    LOG3 plateau a in [0, 1] and midpoint f0 in [-0.5, 1.0] (see _LOG3_BOUNDS).

Sensitivity: a fractional logit on the continuous score (GLM Binomial with the score as the
response), and the P1 conditions recomputed at tau plus and minus 0.1.

Implementation notes (M3):

- The primary MLE is statsmodels GLM at run level with its default cluster correction
  (G/(G-1) * (N-1)/(N-K)) and normal-based Wald limits. The primary Firth fit is
  firthmodels at run level.
- Bootstrap refits use ``newton_logit`` on per-document, per-cell counts. The P1 covariates
  vary only by cell, so the binomial likelihood (and Firth's penalty, whose Fisher
  information sums over runs with equal covariates) on counts equals the run-level one; a
  test checks both against the library fits. One Newton pass fits all 2,000 resamples.
- Each resample goes through the same ladder as the primary fit, because the ladder is
  the estimator: MLE, then Firth when the MLE fails the separation rule (non-convergence or
  any |b_j| above the cap), then failed when Firth does not converge either. The share of
  resamples refitted by Firth is ``firth_fraction``; the failed share is
  ``failed_fraction`` (r4 p1.bootstrap.refit and failed_resamples).
- A resample with no peak (b2 >= 0, or a failed refit) enters the t* percentiles as -inf when
  b1 <= 0 (the fitted zero-row curve is higher at 2 tools than at 6) and as +inf when b1 > 0.
  The design is symmetric about 4 tools, so f(6) - f(2) = 4 b1. Such draws are counted, not
  dropped (r4 p1.bootstrap.no_peak_resamples).
- Coefficient intervals from the bootstrap (the decision intervals in the Firth rung) use
  the successful resamples only; their share is reported.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln
from statsmodels.stats.proportion import proportion_confint

from islands_harness.config import PreRegistration
from islands_harness.stats.resample import (
    percentile_interval,
    resample_indices,
    resample_record,
    resample_weights,
    write_resamples,
)

COEFS = ("const", "tc", "tc2", "f")

Z95 = 1.959964


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval from statsmodels. (0, 0) when n == 0."""
    if n <= 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return (float(lo), float(hi))


def newcombe(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe (1998) method 10 interval for p1 - p2 from the two Wilson limits."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, alpha)
    l2, u2 = wilson(k2, n2, alpha)
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lower, upper)


@dataclass(frozen=True)
class CellStat:
    tools: int
    fault: float
    n: int
    k: int
    p: float
    lo: float
    hi: float
    in_scope: bool
    mean_score: float | None
    faults_per_run: float | None


def cells_table(runs: list[dict[str, Any]], *, alpha: float = 0.05) -> pd.DataFrame:
    """Per-cell counts, rates and Wilson intervals from runs.jsonl rows.

    Rows with ``success`` None (aborted_transport) are excluded from n. Columns:
    tools, fault_rate, n, k, p, lo, hi, in_scope, mean_score, faults_per_run.
    """
    rows = [r for r in runs if r.get("success") is not None]
    mixes = {r.get("mix") for r in rows}
    if len(mixes) > 1:
        raise ValueError(
            f"cells_table pools cells across mixes {sorted(map(str, mixes))}; filter first"
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "tools",
                "fault_rate",
                "n",
                "k",
                "p",
                "lo",
                "hi",
                "in_scope",
                "mean_score",
                "faults_per_run",
            ]
        )
    df["n_faults"] = df["fault_events"].apply(lambda v: len(v or [])) if "fault_events" in df else 0
    g = df.groupby(["tools", "fault_rate"], as_index=False).agg(
        n=("success", "size"),
        k=("success", "sum"),
        mean_score=("score", "mean"),
        faults_per_run=("n_faults", "mean"),
    )
    g["p"] = g["k"] / g["n"]
    bounds = [wilson(int(k), int(n), alpha) for k, n in zip(g["k"], g["n"], strict=True)]
    g["lo"] = [b[0] for b in bounds]
    g["hi"] = [b[1] for b in bounds]
    g["in_scope"] = g["tools"] >= 2
    return g.sort_values(["fault_rate", "tools"]).reset_index(drop=True)


def cell_stats(table: pd.DataFrame) -> list[CellStat]:
    return [
        CellStat(
            int(r.tools),
            float(r.fault_rate),
            int(r.n),
            int(r.k),
            float(r.p),
            float(r.lo),
            float(r.hi),
            bool(r.in_scope),
            None if pd.isna(r.mean_score) else float(r.mean_score),
            None if pd.isna(r.faults_per_run) else float(r.faults_per_run),
        )
        for r in table.itertuples(index=False)
    ]


@dataclass(frozen=True)
class FitResult:
    status: Literal["mle", "firth", "not_estimable"]
    coefficients: dict[str, float]  # b0 (const), tc, tc2, f
    wald_intervals: dict[str, tuple[float, float]]  # cluster-robust Wald, MLE only
    converged: bool
    max_abs_coefficient: float
    n_runs: int
    n_documents: int
    reason: str | None = None  # why the ladder moved, or why not estimable
    notes: tuple[str, ...] = ()  # library warnings, recorded but deciding nothing


@dataclass(frozen=True)
class BootstrapResult:
    draws: int
    peak_interval: tuple[float, float] | None
    drop_interval: tuple[float, float] | None  # in percentage points, descriptive
    b2_negative_fraction: float
    resamples_path: str | None
    coef_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    failed_fraction: float = 0.0  # resamples whose Firth refit failed as well
    rung: str = ""  # the primary fit's rung
    firth_fraction: float = 0.0  # resamples whose MLE failed the separation rule


def peak(b1: float, b2: float) -> float | None:
    """t* = 4 - b1 / (2 b2), defined only when b2 < 0."""
    if b2 >= 0:
        return None
    return 4.0 - b1 / (2.0 * b2)


def design_matrix(runs: list[dict[str, Any]], prereg: PreRegistration) -> pd.DataFrame:
    """In-scope rows with columns y (success as 0/1), score, doc_id, const, tc, tc2, f."""
    scope = prereg.p1.scope
    rows = [
        r
        for r in runs
        if r.get("mix") == scope.mix
        and r.get("tools") in scope.tools
        and r.get("success") is not None
        and r.get("outcome") not in scope.outcomes_excluded
        and r.get("phase", "sweep") == "sweep"
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["const"] = 1.0
    df["tc"] = df["tools"].astype(float) - 4.0
    df["tc2"] = df["tc"] ** 2
    df["f"] = df["fault_rate"].astype(float)
    df["y"] = df["success"].astype(int)
    df["score"] = (
        pd.to_numeric(df["score"], errors="coerce") if "score" in df else df["y"].astype(float)
    )
    return df[["y", "score", "doc_id", "const", "tc", "tc2", "f"]]


def _not_estimable(n_runs: int, n_docs: int, reason: str) -> FitResult:
    return FitResult("not_estimable", {}, {}, False, float("nan"), n_runs, n_docs, reason)


def _mle_cluster(design: pd.DataFrame, alpha: float) -> FitResult:
    """Run-level binomial GLM with document-clustered covariance (statsmodels defaults)."""
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

    x = design[list(COEFS)].to_numpy(float)
    y = design["y"].to_numpy(float)
    groups = pd.factorize(design["doc_id"], sort=True)[0]
    n_docs = int(design["doc_id"].nunique())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = sm.GLM(y, x, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": groups}
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, ZeroDivisionError) as exc:
            return FitResult(
                "mle", {}, {}, False, float("inf"), len(y), n_docs, f"MLE failed: {exc}"
            )
    separation = any(issubclass(w.category, PerfectSeparationWarning) for w in caught)
    notes = ("statsmodels reported PerfectSeparationWarning",) if separation else ()
    params = np.asarray(res.params, dtype=float)
    ci = np.asarray(res.conf_int(alpha), dtype=float)
    finite = bool(np.isfinite(params).all() and np.isfinite(ci).all())
    return FitResult(
        status="mle",
        coefficients={k: float(params[i]) for i, k in enumerate(COEFS)},
        wald_intervals={k: (float(ci[i, 0]), float(ci[i, 1])) for i, k in enumerate(COEFS)},
        converged=bool(getattr(res, "converged", True)) and finite,
        max_abs_coefficient=float(np.max(np.abs(params))) if finite else float("inf"),
        n_runs=len(y),
        n_documents=n_docs,
        notes=notes,
    )


def firth_fit(design: pd.DataFrame) -> FitResult:
    """Firth's penalized likelihood via firthmodels.FirthLogisticRegression on (tc, tc2, f)
    with fit_intercept=True and max_iter=100. Intervals come from the cluster bootstrap, not
    from Firth's Wald standard errors, so ``wald_intervals`` is empty here."""
    from firthmodels import FirthLogisticRegression

    x = design[["tc", "tc2", "f"]].to_numpy(float)
    y = design["y"].to_numpy(int)
    n_docs = int(design["doc_id"].nunique())
    try:
        model = FirthLogisticRegression(max_iter=100).fit(x, y)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeError) as exc:
        return FitResult(
            "firth", {}, {}, False, float("inf"), len(y), n_docs, f"Firth failed: {exc}"
        )
    params = np.array([float(model.intercept_), *np.asarray(model.coef_, dtype=float)])
    finite = bool(np.isfinite(params).all())
    return FitResult(
        status="firth",
        coefficients={k: float(params[i]) for i, k in enumerate(COEFS)},
        wald_intervals={},
        converged=bool(model.converged_) and finite,
        max_abs_coefficient=float(np.max(np.abs(params))) if finite else float("inf"),
        n_runs=len(y),
        n_documents=n_docs,
    )


def fit_p1(runs: list[dict[str, Any]], prereg: PreRegistration) -> FitResult:
    """The fit ladder: MLE with cluster-robust covariance; Firth when the MLE fails the
    separation rule (non-convergence, or any |b_j| above the cap); not_estimable when the
    in-scope outcome has no variation, when the in-scope cells cannot identify the four
    coefficients (a partial sweep), or when Firth fails too. A saturated column never moves
    the ladder by itself: only the fit diagnostics do."""
    design = design_matrix(runs, prereg)
    if design.empty:
        return _not_estimable(0, 0, "no in-scope runs")
    n_runs, n_docs = len(design), int(design["doc_id"].nunique())
    if design["y"].nunique() < 2:
        which = "a success" if int(design["y"].iloc[0]) == 1 else "a failure"
        return _not_estimable(n_runs, n_docs, f"no variation in scope: every run is {which}")
    if n_docs < 2:
        return _not_estimable(
            n_runs, n_docs, "one document: a covariance clustered by document needs two or more"
        )
    rank = int(np.linalg.matrix_rank(design[list(COEFS)].drop_duplicates().to_numpy(float)))
    if rank < len(COEFS):
        return _not_estimable(
            n_runs,
            n_docs,
            f"the in-scope cells identify {rank} of {len(COEFS)} coefficients "
            "(a partial sweep needs at least three tool columns and two fault rows)",
        )
    rule = prereg.p1.separation_rule
    mle = _mle_cluster(design, prereg.alpha)
    reasons = []
    if rule.nonconvergence and not mle.converged:
        reasons.append(mle.reason or "MLE did not converge")
    if mle.max_abs_coefficient > rule.max_abs_coefficient:
        reasons.append(
            f"max |b_j| = {mle.max_abs_coefficient:.3g} exceeds {rule.max_abs_coefficient:g}"
        )
    if not reasons:
        return mle
    if "firth_bootstrap" not in prereg.p1.ladder:
        return _not_estimable(n_runs, n_docs, "; ".join(reasons))
    firth = firth_fit(design)
    if firth.converged:
        return replace(firth, reason="; ".join(reasons), notes=mle.notes)
    failure = firth.reason or "Firth did not converge"
    return _not_estimable(n_runs, n_docs, "; ".join([*reasons, failure]))


@dataclass(frozen=True)
class Aggregate:
    """Per-document, per-cell counts: the sufficient statistics of the P1 model."""

    x: np.ndarray  # (C, 4): const, tc, tc2, f per distinct cell
    s: np.ndarray  # (D, C) successes
    n: np.ndarray  # (D, C) runs
    documents: tuple[str, ...]
    cells: tuple[tuple[int, float], ...]  # (tools, fault) per column


def aggregate(design: pd.DataFrame) -> Aggregate:
    """Collapse a design frame to counts per (document, cell)."""
    docs = tuple(sorted(str(d) for d in design["doc_id"].unique()))
    pairs = list(zip(design["tc"].astype(float), design["f"].astype(float), strict=True))
    keys = sorted(set(pairs))
    col = {k: i for i, k in enumerate(keys)}
    d_code = pd.Categorical(design["doc_id"].astype(str), categories=list(docs)).codes
    c_code = np.array([col[k] for k in pairs])
    s = np.zeros((len(docs), len(keys)))
    n = np.zeros((len(docs), len(keys)))
    np.add.at(s, (d_code, c_code), design["y"].to_numpy(float))
    np.add.at(n, (d_code, c_code), 1.0)
    x = np.array([[1.0, tc, tc * tc, f] for tc, f in keys])
    cells = tuple((int(round(tc + 4.0)), f) for tc, f in keys)
    return Aggregate(x, s, n, docs, cells)


def newton_logit(
    x: np.ndarray,
    s: np.ndarray,
    n: np.ndarray,
    *,
    firth: bool = False,
    max_iter: int = 100,
    tol: float = 1e-8,
    max_step: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Binomial-logit Newton-Raphson on counts, vectorized over a batch of datasets.

    x is (C, K); s and n are (B, C). Returns (beta (B, K), converged (B,)). With ``firth``
    the score is Firth's modified score X'(s - n p + h (1/2 - p)), h the hat diagonal of
    W^(1/2) X (X'WX)^-1 X' W^(1/2) with W = n p (1 - p), and the step uses the Fisher
    information (Heinze and Schemper 2002). Steps are capped at ``max_step`` in the largest
    coordinate. A dataset converges when its largest step is below ``tol``.
    """
    s = np.atleast_2d(np.asarray(s, dtype=float))
    n = np.atleast_2d(np.asarray(n, dtype=float))
    batch, k = s.shape[0], x.shape[1]
    beta = np.zeros((batch, k))
    converged = np.zeros(batch, dtype=bool)
    active = np.ones(batch, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        b, sb, nb = beta[idx], s[idx], n[idx]
        p = expit(b @ x.T)
        w = nb * p * (1.0 - p)
        info = np.einsum("bc,ci,cj->bij", w, x, x)
        try:
            inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(info)
        r = sb - nb * p
        if firth:
            h = w * np.einsum("ci,bij,cj->bc", x, inv, x)
            r = r + h * (0.5 - p)
        step = np.einsum("bij,bj->bi", inv, r @ x)
        big = np.abs(step).max(axis=1)
        scale = np.where(big > max_step, max_step / np.maximum(big, 1e-300), 1.0)
        new = b + step * scale[:, None]
        bad = ~np.isfinite(new).all(axis=1)
        beta[idx] = np.where(bad[:, None], b, new)
        active[idx[bad]] = False
        done = ~bad & (big < tol)
        converged[idx[done]] = True
        active[idx[done]] = False
    return beta, converged


def peak_draws(beta: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """t* per resample; a resample without a peak maps to -inf (b1 <= 0) or +inf (b1 > 0)."""
    b1, b2 = beta[:, 1], beta[:, 2]
    has_peak = ok & (b2 < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 4.0 - b1 / (2.0 * b2)
    side = np.where(b1 > 0, np.inf, -np.inf)
    return np.where(has_peak, t, side)


def drop_column(t_star: float | None, prereg: PreRegistration) -> int:
    """The column C3 compares with six tools: t* rounded half up and clipped to the peak
    range, or the
    lowest in-scope rung when there is no peak (b2 >= 0), so the drop is then measured across
    the whole range (r4 p1.drop.peak_column)."""
    lo, hi = prereg.p1.peak.range
    if t_star is None:
        return int(lo)
    if t_star <= lo:
        return int(lo)
    if t_star >= hi:
        return int(hi)
    return int(math.floor(t_star + 0.5))  # half up, so every implementation agrees on ties


def cluster_bootstrap(
    runs: list[dict[str, Any]],
    prereg: PreRegistration,
    draws: int,
    *,
    snapshot_id: str,
    model_id: str,
    out_path: str | None,
    fit: FitResult | None = None,
    statistic: str = "p1",
) -> BootstrapResult:
    """Document-cluster percentile bootstrap of t*, of every coefficient and of the zero-row
    drop, refitting each resample in the primary fit's rung.

    Resample d_j = documents[u32("boot", snapshot_id, model_id, statistic, draw, j) % D] for
    j in range(D) over the sorted in-scope documents; refit on all runs of the resampled
    documents (a document drawn twice counts twice) through the fit ladder. The drop is p(drop column) - p(6) in the
    zero row with the column fixed from the primary fit (drop_column); it is descriptive (C3
    uses the Newcombe interval). Writes {"statistics": {statistic: {"documents", "draws", "indices"}}}
    to ``out_path`` when given.
    """
    rule = prereg.p1.bootstrap
    if (rule.refit, rule.no_peak_resamples, rule.failed_resamples) != (
        "ladder_per_resample",
        "signed_infinity_by_slope",
        "counted_as_no_peak",
    ):
        raise ValueError(
            "r4 p1.bootstrap must name refit: ladder_per_resample, no_peak_resamples: "
            "signed_infinity_by_slope and failed_resamples: counted_as_no_peak, the rules "
            "this implementation applies"
        )
    fit = fit if fit is not None else fit_p1(runs, prereg)
    design = design_matrix(runs, prereg)
    if fit.status == "not_estimable" or design.empty:
        return BootstrapResult(draws, None, None, 0.0, None, rung=fit.status)
    agg = aggregate(design)
    idx = resample_indices(snapshot_id, model_id, statistic, draws, len(agg.documents))
    weights = resample_weights(idx)
    s_b, n_b = weights @ agg.s, weights @ agg.n
    sep = prereg.p1.separation_rule
    beta, converged = newton_logit(agg.x, s_b, n_b)
    mle_ok = np.isfinite(beta).all(axis=1) & (np.abs(beta).max(axis=1) <= sep.max_abs_coefficient)
    if sep.nonconvergence:
        mle_ok &= converged
    ok = mle_ok.copy()
    redo = ~mle_ok
    if redo.any() and "firth_bootstrap" in prereg.p1.ladder:
        f_beta, f_conv = newton_logit(agg.x, s_b[redo], n_b[redo], firth=True)
        beta[redo] = f_beta
        ok[redo] = f_conv & np.isfinite(f_beta).all(axis=1)
    else:
        ok[redo] = False
    alpha = prereg.alpha
    peak_iv = percentile_interval(peak_draws(beta, ok), alpha)
    coef_ivs = (
        {k: percentile_interval(beta[ok, i], alpha) for i, k in enumerate(COEFS)}
        if ok.any()
        else {}
    )
    drop_iv = None
    col = drop_column(peak(fit.coefficients["tc"], fit.coefficients["tc2"]), prereg)
    row, last = float(prereg.p1.drop.row), int(prereg.p1.peak.range[1])
    if (col, row) in agg.cells and (last, row) in agg.cells:
        a, b = agg.cells.index((col, row)), agg.cells.index((last, row))
        valid = (n_b[:, a] > 0) & (n_b[:, b] > 0)
        if valid.any():
            pa = s_b[valid, a] / n_b[valid, a]
            pb = s_b[valid, b] / n_b[valid, b]
            drop_iv = percentile_interval(100.0 * (pa - pb), alpha)
    if out_path is not None:
        write_resamples(
            out_path,
            snapshot_id=snapshot_id,
            model_id=model_id,
            statistics={statistic: resample_record(list(agg.documents), idx)},
        )
    return BootstrapResult(
        draws=draws,
        peak_interval=peak_iv,
        drop_interval=drop_iv,
        b2_negative_fraction=float(np.mean(ok & (beta[:, 2] < 0))),
        resamples_path=out_path,
        coef_intervals=coef_ivs,
        failed_fraction=float(np.mean(~ok)),
        rung=fit.status,
        firth_fraction=float(np.mean(redo)),
    )


def fault_cap_check(
    fit: FitResult, intervals: dict[str, tuple[float, float]] | None = None
) -> bool:
    """True when b3 <= 0 or its interval includes 0. A positive fault effect with an interval
    excluding 0 means faults help, which is a harness anomaly, not a finding. ``intervals``
    are the rung's decision intervals (Wald for MLE, bootstrap for Firth); the fit's Wald
    intervals by default."""
    b3 = fit.coefficients.get("f")
    if b3 is None:
        return True
    if b3 <= 0:
        return True
    source = fit.wald_intervals if intervals is None else intervals
    lo, hi = source.get("f", (float("-inf"), float("inf")))
    return lo <= 0 <= hi


@dataclass(frozen=True)
class ShapeResult:
    column: int | None  # None for pooled
    aic_lin: float
    aic_log3: float
    f0: float
    s: float
    label: Literal["cliff", "slope", "indistinguishable"]
    a: float = float("nan")  # LOG3 plateau
    lin_intercept: float = float("nan")
    lin_slope: float = float("nan")


_EPS = 1e-6
_LOG3_GRID = [
    (a, s, f0)
    for a in (0.5, 0.9, 1.0)
    for s in (5.0, 10.0, 20.0, 40.0)
    for f0 in (0.1, 0.2, 0.3, 0.4)
]
_LIN_GRID = [(a, b) for a in (0.5, 0.9, 1.0) for b in (-2.0, -1.0, 0.0)]


def _binomial_loglik(p: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    """Full binomial log-likelihood, constant included, with p clipped to [eps, 1 - eps]."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    const = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    return float(np.sum(const + k * np.log(p) + (n - k) * np.log1p(-p)))


def _lin(theta: np.ndarray, f: np.ndarray) -> np.ndarray:
    return theta[0] + theta[1] * f


def _log3(theta: np.ndarray, f: np.ndarray) -> np.ndarray:
    # a / (1 + exp(s (f - f0))), written through expit so large s cannot overflow
    return theta[0] * expit(-theta[1] * (f - theta[2]))


# Parameter bounds for the bounded Nelder-Mead fits. LIN: intercept and slope (the clip keeps p
# inside (0, 1)). LOG3: plateau a, steepness s, midpoint f0. The plateau is a probability, so
# it lies in [0, 1], and the midpoint lies within half a fault range of the observed rates.
# Unbounded, the simplex drifted to plateaus in the thousands with midpoints far left of the
# data, where a / (1 + exp(s (f - f0))) is an exponential decay rather than a step. With its
# extra parameter that decay imitated a straight decline, so straight declines came out
# "indistinguishable" instead of "slope" (56 of 120 simulated columns).
_LIN_BOUNDS = [(-1.0, 2.0), (-20.0, 20.0)]
_LOG3_BOUNDS = [(0.0, 1.0), (-200.0, 200.0), (-0.5, 1.0)]


def _best_fit(
    model: Any,
    starts: list[tuple[float, ...]],
    bounds: list[tuple[float, float]],
    f: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Maximum likelihood by bounded Nelder-Mead from every start of a fixed grid; the first
    start reaching the lowest negative log-likelihood wins, so the result is deterministic."""
    from scipy.optimize import minimize

    lower, upper = [b[0] for b in bounds], [b[1] for b in bounds]

    def nll(th: np.ndarray) -> float:
        return -_binomial_loglik(model(th, f), k, n)

    best_x, best_nll = None, math.inf
    for x0 in starts:
        res = minimize(
            nll,
            np.clip(np.asarray(x0, dtype=float), lower, upper),
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000, "maxfev": 4000},
        )
        if res.fun < best_nll - 1e-12:
            best_x, best_nll = np.asarray(res.x, dtype=float), float(res.fun)
    assert best_x is not None
    return best_x, best_nll


def _shape_one(column: int | None, f: np.ndarray, k: np.ndarray, n: np.ndarray) -> ShapeResult:
    rates = k / n
    slope, intercept = np.polyfit(f, rates, 1, w=np.sqrt(n))
    lin_starts = [(float(intercept), float(slope)), *_LIN_GRID]
    lin_x, lin_nll = _best_fit(_lin, lin_starts, _LIN_BOUNDS, f, k, n)
    log_x, log_nll = _best_fit(_log3, list(_LOG3_GRID), _LOG3_BOUNDS, f, k, n)
    aic_lin = 2 * 2 + 2 * lin_nll
    aic_log3 = 2 * 3 + 2 * log_nll
    diff = aic_lin - aic_log3
    label: Literal["cliff", "slope", "indistinguishable"] = (
        "cliff" if diff >= 2 else "slope" if diff <= -2 else "indistinguishable"
    )
    return ShapeResult(
        column=column,
        aic_lin=aic_lin,
        aic_log3=aic_log3,
        f0=float(log_x[2]),
        s=float(log_x[1]),
        label=label,
        a=float(log_x[0]),
        lin_intercept=float(lin_x[0]),
        lin_slope=float(lin_x[1]),
    )


def shape_fit(cells: list[CellStat]) -> list[ShapeResult]:
    """LIN versus LOG3 by AIC per in-scope column and pooled, per the module docstring.
    Columns with fewer than four fault levels are skipped (LOG3 has three parameters)."""
    scope = [c for c in cells if c.in_scope and c.n > 0]
    out: list[ShapeResult] = []
    for tools in sorted({c.tools for c in scope}):
        col = sorted((c for c in scope if c.tools == tools), key=lambda c: c.fault)
        if len(col) < 4:
            continue
        f = np.array([c.fault for c in col])
        out.append(
            _shape_one(
                tools, f, np.array([c.k for c in col], float), np.array([c.n for c in col], float)
            )
        )
    faults = sorted({c.fault for c in scope})
    if len(faults) >= 4:
        k = np.array([sum(c.k for c in scope if c.fault == f) for f in faults], float)
        n = np.array([sum(c.n for c in scope if c.fault == f) for f in faults], float)
        out.append(_shape_one(None, np.array(faults), k, n))
    return out


def _fractional_logit(runs: list[dict[str, Any]], prereg: PreRegistration) -> dict[str, Any]:
    """GLM Binomial on the continuous score with document-clustered covariance. Descriptive."""
    import statsmodels.api as sm

    design = design_matrix(runs, prereg)
    design = design[design["score"].notna()] if not design.empty else design
    if design.empty or design["score"].nunique() < 2 or design["doc_id"].nunique() < 2:
        return {"status": "not_estimable", "reason": "no variation in score, or one document"}
    x = design[list(COEFS)].to_numpy(float)
    groups = pd.factorize(design["doc_id"], sort=True)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(design["score"].to_numpy(float), x, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    params = np.asarray(res.params, dtype=float)
    ci = np.asarray(res.conf_int(prereg.alpha), dtype=float)
    return {
        "status": "fitted",
        "coefficients": {k: float(params[i]) for i, k in enumerate(COEFS)},
        "intervals": {k: [float(ci[i, 0]), float(ci[i, 1])] for i, k in enumerate(COEFS)},
        "peak": peak(float(params[1]), float(params[2])),
    }


def sensitivity(
    runs: list[dict[str, Any]],
    prereg: PreRegistration,
    tau_offsets: tuple[float, ...] = (-0.1, 0.1),
    *,
    draws: int = 2000,
    snapshot_id: str = "",
    model_id: str = "",
) -> dict[str, Any]:
    """P1 conditions recomputed with success = score >= tau + offset (tau rounded to six
    decimals, the same comparison grade_run makes), each with its own fit, bootstrap on the
    same resample indices as the primary analysis, and verdict; plus the fractional logit on
    the continuous score. Descriptive: nothing here moves a verdict."""
    from islands_harness.stats.verdict import verdict_p1

    out: dict[str, Any] = {"descriptive": True, "tau": []}
    for offset in tau_offsets:
        tau = round(prereg.success.tau + offset, 6)
        rows = [
            r
            if r.get("success") is None or r.get("score") is None
            else {**r, "success": float(r["score"]) >= tau}
            for r in runs
        ]
        fit = fit_p1(rows, prereg)
        boot = cluster_bootstrap(
            rows, prereg, draws, snapshot_id=snapshot_id, model_id=model_id, out_path=None, fit=fit
        )
        verdict = verdict_p1(fit, boot, cell_stats(cells_table(rows)), prereg)
        out["tau"].append(
            {
                "tau": tau,
                "offset": offset,
                "fit_status": fit.status,
                "verdict": verdict.verdict.value,
                "peak": verdict.peak,
                "peak_interval": verdict.peak_interval,
                "drop_points": verdict.drop_points,
                "drop_interval": verdict.drop_interval,
                "conditions": [
                    {"name": c.name, "value": c.value, "passed": c.passed}
                    for c in verdict.conditions
                ],
            }
        )
    out["fractional_logit"] = _fractional_logit(runs, prereg)
    return out


def viable_fraction(cells: list[CellStat]) -> dict[str, float]:
    """Share of in-scope cells with p > 0.5, and the share whose Wilson lower bound is > 0.5."""
    scope = [c for c in cells if c.in_scope]
    if not scope:
        return {"point": 0.0, "lower_bound": 0.0, "n_cells": 0}
    return {
        "point": sum(c.p > 0.5 for c in scope) / len(scope),
        "lower_bound": sum(c.lo > 0.5 for c in scope) / len(scope),
        "n_cells": len(scope),
    }


def boundary_width(cells: list[CellStat], alpha: float = 0.05) -> dict[str, Any]:
    """Per fault row: the rungs between the first in-scope cell below 90 percent and the first
    below 10 percent along the tool axis. The width is claimed only when every adjacent
    Newcombe interval on the way excludes 0, the way running from the last cell at or above
    90 percent (or the first cell, if none is) to the first cell below 10 percent. A row that
    never falls below 10 percent has no width. Descriptive."""
    scope = [c for c in cells if c.in_scope and c.n > 0]
    rows: list[dict[str, Any]] = []
    for fault in sorted({c.fault for c in scope}):
        line = sorted((c for c in scope if abs(c.fault - fault) < 1e-9), key=lambda c: c.tools)
        i90 = next((i for i, c in enumerate(line) if c.p < 0.9), None)
        i10 = next((i for i, c in enumerate(line) if c.p < 0.1), None)
        entry: dict[str, Any] = {
            "fault": fault,
            "first_below_90": None if i90 is None else line[i90].tools,
            "first_below_10": None if i10 is None else line[i10].tools,
            "width": None,
            "claimed": False,
        }
        if i90 is not None and i10 is not None:
            entry["width"] = line[i10].tools - line[i90].tools
            steps = [
                newcombe(line[j].k, line[j].n, line[j + 1].k, line[j + 1].n, alpha)
                for j in range(max(i90 - 1, 0), i10)
            ]
            entry["claimed"] = bool(steps) and all(lo > 0 or hi < 0 for lo, hi in steps)
        rows.append(entry)
    return {"descriptive": True, "rows": rows}
