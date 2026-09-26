"""Statistics verified against known truth and against the site's illustrative tables.

The fixtures FIGURE_A (a band: high success at two to four tools, falling at five and six)
and FIGURE_B (a continent: six tools within sampling error of the peak) are the illustrative
grids on the site's Map page, percent success at n = 100 per cell, tools 2..6 across and
fault rates 0..50 down. The verdict on A must be ``survives`` and on B ``restricted``.

Coverage and false-survival assertions run on a reduced snapshot count by default
(ISLANDS_SIMULATION_SNAPSHOTS, 40 in CI); the full 500 per truth runs behind
``-m full_simulation``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from islands_harness.config import load_prereg
from islands_harness.stats import simulate
from islands_harness.stats.fit import (
    BootstrapResult,
    CellStat,
    FitResult,
    cell_stats,
    cells_table,
    newcombe,
    peak,
    wilson,
)
from islands_harness.stats.recovery import RecoveryResult
from islands_harness.stats.verdict import (
    P1Verdict,
    P2Verdict,
    Verdict,
    instrument_check,
    statement,
    verdict_p1,
    verdict_p2,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = load_prereg(REPO / "configs" / "preregistration" / "r4.yaml")
N_SNAPSHOTS = int(os.environ.get("ISLANDS_SIMULATION_SNAPSHOTS", "40"))

FAULTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
FIGURE_A = [  # rows: fault 0..50; columns: tools 2..6
    [88, 93, 90, 61, 33],
    [85, 90, 86, 56, 30],
    [74, 81, 75, 47, 25],
    [52, 58, 53, 35, 20],
    [30, 33, 31, 24, 15],
    [18, 19, 18, 15, 11],
]
FIGURE_B = [
    [90, 92, 91, 89, 88],
    [84, 86, 85, 83, 82],
    [72, 74, 73, 71, 70],
    [55, 57, 56, 54, 53],
    [36, 37, 36, 35, 34],
    [21, 22, 21, 20, 20],
]


def _cells(table: list[list[int]], n: int = 100) -> list[CellStat]:
    cells = []
    for row, fault in zip(table, FAULTS, strict=True):
        for tools, pct in zip(range(2, 7), row, strict=True):
            k = round(pct * n / 100)
            lo, hi = wilson(k, n)
            cells.append(CellStat(tools, fault, n, k, k / n, lo, hi, True, None, None))
    return cells


# -- intervals ------------------------------------------------------------------------------


def _wilson_closed_form(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    # z is the exact 97.5th normal percentile (ARCHITECTURE.md Section 9); a rounded z
    # moves the limits by about 3e-9, more than the tolerance below.
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


@pytest.mark.parametrize("k,n", [(50, 100), (0, 10), (10, 10), (93, 100), (1, 50), (25, 50)])
def test_wilson_matches_closed_form(k: int, n: int) -> None:
    lo, hi = wilson(k, n)
    elo, ehi = _wilson_closed_form(k, n)
    assert lo == pytest.approx(elo, abs=1e-9)
    assert hi == pytest.approx(ehi, abs=1e-9)


def test_wilson_published_values() -> None:
    assert wilson(50, 100) == pytest.approx((0.4038, 0.5962), abs=1e-3)
    assert wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-3)
    lo, hi = wilson(50, 100)
    assert (hi - lo) / 2 == pytest.approx(0.097, abs=2e-3)  # about 9.7 points at n = 100 near 0.5


def test_newcombe_properties() -> None:
    lo, hi = newcombe(93, 100, 33, 100)
    assert lo < 0.60 < hi and lo > 0
    rlo, rhi = newcombe(33, 100, 93, 100)
    assert (rlo, rhi) == pytest.approx((-hi, -lo))
    slo, shi = newcombe(50, 100, 50, 100)
    assert slo == pytest.approx(-shi) and slo < 0 < shi


# -- verdicts on the site's tables ------------------------------------------------------------


def _fit(
    b1: float, b2: float, b2_iv: tuple[float, float], b3: float = -4.0, status: str = "mle"
) -> FitResult:
    return FitResult(
        status=status,  # type: ignore[arg-type]
        coefficients={"const": 1.5, "tc": b1, "tc2": b2, "f": b3},
        wald_intervals={
            "const": (1.0, 2.0),
            "tc": (b1 - 0.3, b1 + 0.3),
            "tc2": b2_iv,
            "f": (b3 - 1, b3 + 1),
        },
        converged=True,
        max_abs_coefficient=max(abs(b1), abs(b2), abs(b3), 1.5),
        n_runs=3000,
        n_documents=100,
    )


def test_table_a_survives() -> None:
    fit = _fit(b1=-0.7, b2=-0.35, b2_iv=(-0.5, -0.2))
    assert peak(-0.7, -0.35) == pytest.approx(3.0)
    boot = BootstrapResult(2000, (2.5, 3.6), None, 1.0, None)
    v = verdict_p1(fit, boot, _cells(FIGURE_A), PREREG)
    assert v.verdict == Verdict.survives
    assert {c.name: c.passed for c in v.conditions if c.name.startswith("C")} == {
        "C1_curvature": True,
        "C2_peak": True,
        "C3_drop": True,
    }
    assert v.drop_points == pytest.approx(60.0)
    assert v.drop_interval is not None and v.drop_interval[0] > 0


def test_table_b_restricts() -> None:
    fit = _fit(b1=0.0, b2=-0.01, b2_iv=(-0.05, 0.03))
    boot = BootstrapResult(2000, (1.0, 9.0), None, 0.55, None)
    v = verdict_p1(fit, boot, _cells(FIGURE_B), PREREG)
    assert v.verdict == Verdict.restricted
    assert v.drop_interval is not None and v.drop_interval[1] < PREREG.p1.drop.sesoi_points


def test_not_estimable_is_its_own_category() -> None:
    fit = _fit(0.0, 0.0, (0.0, 0.0), status="not_estimable")
    v = verdict_p1(fit, None, _cells(FIGURE_A), PREREG)
    assert v.verdict == Verdict.not_estimable
    assert v.conditions[0].name == "estimable" and v.conditions[0].passed is False


def test_sesoi_equivalence_is_below_resolution() -> None:
    small = [
        CellStat(t, 0.0, 30, k, k / 30, *wilson(k, 30), True, None, None)
        for t, k in ((2, 17), (3, 18), (4, 17), (5, 16), (6, 15))
    ]
    fit = _fit(b1=-0.7, b2=-0.35, b2_iv=(-0.8, 0.1))
    boot = BootstrapResult(2000, (2.2, 5.5), None, 0.7, None)
    v = verdict_p1(fit, boot, small, PREREG)
    assert v.verdict == Verdict.below_resolution
    assert any(c.name == "sesoi_equivalence" for c in v.conditions)
    assert (
        v.drop_interval is not None
        and v.drop_interval[0] <= 0 <= v.drop_interval[1]
        and v.drop_interval[1] >= 15
    )


def test_fault_cap_failure_blocks() -> None:
    fit = _fit(b1=-0.7, b2=-0.35, b2_iv=(-0.5, -0.2), b3=2.0)
    v = verdict_p1(
        fit, BootstrapResult(2000, (2.5, 3.6), None, 1.0, None), _cells(FIGURE_A), PREREG
    )
    assert v.verdict == Verdict.blocked


def test_instrument_check_uses_band_and_half_width() -> None:
    cells = _cells(FIGURE_A)
    assert (
        instrument_check(cells, PREREG).label == "in_calibration"
    )  # anchor 88 percent, band [0.70, 0.90]
    low = [CellStat(2, 0.0, 100, 5, 0.05, *wilson(5, 100), True, None, None)]
    assert instrument_check(low, PREREG).label == "instrument_out_of_calibration"
    perfect = [CellStat(2, 0.0, 100, 100, 1.0, *wilson(100, 100), True, None, None)]
    assert instrument_check(perfect, PREREG).label == "instrument_out_of_calibration"
    mid = [CellStat(2, 0.0, 100, 40, 0.4, *wilson(40, 100), True, None, None)]
    assert instrument_check(mid, PREREG).label == "in_calibration"  # inside [0.15, 0.95]
    assert instrument_check([], PREREG).label == "anchor_missing"


def _recovery(r: float, lo: float, hi: float) -> RecoveryResult:
    return RecoveryResult(
        r,
        lo,
        hi,
        400,
        90,
        r - 1,
        lo - 1,
        hi - 1,
        2.0,
        {},
        {},
        "exact_invariant_stack",
        "recovered runs only",
    )


def test_p2_categories() -> None:
    assert verdict_p2(_recovery(1.2, 0.4, 2.0), PREREG).verdict == Verdict.survives
    assert verdict_p2(_recovery(0.2, -0.3, 0.7), PREREG).verdict == Verdict.restricted
    assert verdict_p2(_recovery(0.4, -0.6, 1.3), PREREG).verdict == Verdict.below_resolution


def test_statement_format() -> None:
    p1 = P1Verdict(Verdict.survives, [], 3.4, (2.9, 4.1), 41.0, (27.0, 53.0), "mle")
    p2 = P2Verdict(Verdict.below_resolution, [], 0.4, (-0.6, 1.3))
    assert statement("2026-11-15", "s1", "gpt-oss-20b-local", "A", p1, p2) == (
        "2026-11-15 | s1 | gpt-oss-20b-local | mix A | P1: survives (peak 3.4 tools [2.9, 4.1]; "
        "drop 41 points [27, 53]) | P2: below resolution (+0.4 turns [-0.6, 1.3])"
    )
    p1_ne = P1Verdict(Verdict.not_estimable, [], None, None, None, None, "not_estimable")
    assert "P1: not estimable |" in statement("2026-11-15", "s1", "m", "A", p1_ne, p2)


# -- simulation -------------------------------------------------------------------------------


def test_simulated_band_has_the_expected_shape() -> None:
    rows = simulate.simulate_runs(simulate.PRESETS["band"], n_docs=100, epochs=1, scope="t0")
    assert len(rows) == 36 * 100
    table = cells_table(rows)
    cells = {(c.tools, c.fault): c for c in cell_stats(table)}
    assert all(cells[(1, f)].p == 0.0 for f in FAULTS)  # rung 1 scores 0 by construction
    assert cells[(3, 0.0)].p > cells[(6, 0.0)].p + 0.2
    assert cells[(3, 0.0)].p > cells[(3, 0.5)].p
    assert rows == simulate.simulate_runs(
        simulate.PRESETS["band"], n_docs=100, epochs=1, scope="t0"
    )


def test_simulated_continent_is_flat_across_tools() -> None:
    rows = simulate.simulate_runs(simulate.PRESETS["continent"], n_docs=100, epochs=1, scope="t1")
    cells = {(c.tools, c.fault): c for c in cell_stats(cells_table(rows))}
    assert abs(cells[(2, 0.0)].p - cells[(6, 0.0)].p) < 0.15


@pytest.mark.parametrize("n_docs", [100, 50])
def test_coverage_and_false_survival(n_docs: int) -> None:
    band = simulate.coverage_experiment("band", N_SNAPSHOTS, n_docs)
    continent = simulate.coverage_experiment("continent", N_SNAPSHOTS, n_docs)
    flat = simulate.coverage_experiment("flat", N_SNAPSHOTS, n_docs)
    for report in (band, continent, flat):
        assert 0.93 <= report.wilson_coverage <= 0.97
    assert band.peak_coverage is not None and band.peak_coverage >= 0.93
    assert continent.false_survival is not None and continent.false_survival <= 0.05
    assert flat.false_survival is not None and flat.false_survival <= 0.05
    print(f"power under the band at n={n_docs}: {band.power}")  # reported, not asserted


@pytest.mark.full_simulation
@pytest.mark.parametrize("n_docs", [100, 50])
def test_full_coverage_500_snapshots(n_docs: int) -> None:
    for preset in ("band", "continent", "flat"):
        report = simulate.coverage_experiment(preset, 500, n_docs)
        assert 0.93 <= report.wilson_coverage <= 0.97
        if preset == "band":
            assert report.peak_coverage is not None and report.peak_coverage >= 0.93
        else:
            assert report.false_survival is not None and report.false_survival <= 0.05


def test_firth_triggers_on_separation_not_on_saturated_column() -> None:
    from islands_harness.stats.fit import fit_p1

    separated = [
        r for r in simulate.simulate_runs(simulate.PRESETS["band"], 40, 1, "sep") if r["tools"] >= 2
    ]
    for r in separated:  # make success a deterministic function of tools: perfect separation
        r["success"] = r["tools"] <= 3
        r["score"] = float(r["success"])
    assert fit_p1(separated, PREREG).status == "firth"
    saturated = [
        r
        for r in simulate.simulate_runs(simulate.PRESETS["band"], 100, 1, "sat")
        if r["tools"] >= 2
    ]
    for r in saturated:  # one column at 100 percent among mixed cells: MLE stays
        if r["tools"] == 3 and r["fault_rate"] == 0.0:
            r["success"], r["score"] = True, 1.0
    assert fit_p1(saturated, PREREG).status == "mle"


def test_bootstrap_resample_indices_are_written_and_reproducible(tmp_path: Path) -> None:
    from islands_harness.stats.fit import cluster_bootstrap

    rows = simulate.simulate_runs(simulate.PRESETS["band"], 50, 1, "boot")
    a = cluster_bootstrap(
        rows, PREREG, 50, snapshot_id="s1", model_id="m", out_path=str(tmp_path / "a.json")
    )
    b = cluster_bootstrap(
        rows, PREREG, 50, snapshot_id="s1", model_id="m", out_path=str(tmp_path / "b.json")
    )
    assert a.peak_interval == b.peak_interval
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


# -- M3: fits, bootstraps, recovery, descriptives, analysis ----------------------------------


def _design(rows):  # noqa: ANN001, ANN202
    from islands_harness.stats.fit import design_matrix

    return design_matrix(rows, PREREG)


def test_count_newton_matches_the_library_fits() -> None:
    """The batched refit on per-cell counts gives the same MLE as statsmodels at run level and
    the same Firth estimate as firthmodels, which is what licenses using it in the bootstrap."""
    import numpy as np

    from islands_harness.stats.fit import COEFS, aggregate, firth_fit, fit_p1, newton_logit

    rows = simulate.simulate_runs(simulate.PRESETS["band"], 60, 1, "newton")
    agg = aggregate(_design(rows))
    mle = fit_p1(rows, PREREG)
    beta, ok = newton_logit(agg.x, agg.s.sum(axis=0), agg.n.sum(axis=0))
    assert ok[0] and np.allclose(beta[0], [mle.coefficients[k] for k in COEFS], atol=1e-7)
    firth = firth_fit(_design(rows))
    fbeta, fok = newton_logit(agg.x, agg.s.sum(axis=0), agg.n.sum(axis=0), firth=True)
    assert fok[0] and np.allclose(fbeta[0], [firth.coefficients[k] for k in COEFS], atol=1e-4)


def test_resample_indices_follow_the_published_formula() -> None:
    from islands_harness.rng import u32
    from islands_harness.stats.resample import resample_indices, resample_weights

    idx = resample_indices("s1", "m", "p1", 30, 17)
    assert idx.shape == (30, 17)
    for b, j in ((0, 0), (7, 3), (29, 16)):
        assert idx[b, j] == u32("boot", "s1", "m", "p1", b, j) % 17
    w = resample_weights(idx)
    assert (w.sum(axis=1) == 17).all() and w[7, idx[7, 3]] >= 1


def test_percentile_interval_orders_infinite_draws() -> None:
    import numpy as np

    from islands_harness.stats.resample import percentile_interval

    finite = np.linspace(3.0, 4.0, 1990)
    lo, hi = percentile_interval(np.concatenate([np.full(10, -np.inf), finite]), 0.05)
    assert lo == finite[40] and hi < 4.0  # the 51st draw of 2,000 is the 41st finite one
    lo, _ = percentile_interval(np.concatenate([np.full(60, -np.inf), np.full(1940, 3.5)]), 0.05)
    assert lo == -np.inf  # more than 2.5 percent without a peak opens the lower end
    with pytest.raises(ValueError):
        percentile_interval(np.array([1.0, np.nan]), 0.05)


def test_peak_draws_map_no_peak_by_the_sign_of_b1() -> None:
    import numpy as np

    from islands_harness.stats.fit import peak_draws

    beta = np.array(
        [
            [1.0, -0.5, -0.5, -3.0],  # peak at 3.5
            [1.0, -0.5, 0.1, -3.0],  # convex, best at 2 tools
            [1.0, 0.5, 0.0, -3.0],  # linear rising, best at 6 tools
            [1.0, -0.5, -0.5, -3.0],  # failed refit: treated as no peak
        ]
    )
    ok = np.array([True, True, True, False])
    assert list(peak_draws(beta, ok)) == [3.5, -np.inf, np.inf, -np.inf]


def test_rank_deficient_partial_sweep_is_not_estimable() -> None:
    from islands_harness.stats.fit import fit_p1

    keep = {(2, 0.0), (3, 0.2), (6, 0.5)}
    rows = [
        r
        for r in simulate.simulate_runs(simulate.PRESETS["band"], 40, 1, "rank")
        if (r["tools"], r["fault_rate"]) in keep
    ]
    fit = fit_p1(rows, PREREG)
    assert fit.status == "not_estimable" and "identify 3 of 4" in (fit.reason or "")


def test_no_variation_is_not_estimable() -> None:
    from islands_harness.stats.fit import fit_p1

    rows = simulate.simulate_runs(simulate.PRESETS["band"], 20, 1, "flatline")
    for r in rows:
        r["success"], r["score"] = False, 0.0
    fit = fit_p1(rows, PREREG)
    assert fit.status == "not_estimable" and "no variation" in (fit.reason or "")


def test_firth_rung_decides_with_bootstrap_intervals() -> None:
    base = _fit(b1=-0.7, b2=-0.35, b2_iv=(-9.0, 9.0), status="firth")
    fit = FitResult(**{**base.__dict__, "wald_intervals": {}})
    boot = BootstrapResult(
        2000,
        (2.5, 3.6),
        None,
        1.0,
        None,
        coef_intervals={"tc2": (-0.5, -0.2), "f": (-5.0, -3.0)},
    )
    assert verdict_p1(fit, boot, _cells(FIGURE_A), PREREG).verdict == Verdict.survives
    without = BootstrapResult(2000, (2.5, 3.6), None, 1.0, None)
    assert verdict_p1(fit, without, _cells(FIGURE_A), PREREG).verdict != Verdict.survives


def test_no_peak_measures_the_drop_from_two_tools() -> None:
    """With b2 >= 0 there is no peak; C3 fails, but the drop from 2 to 6 tools is still
    measured, so a flat curve measured precisely is restricted rather than unresolved."""
    fit = _fit(b1=0.0, b2=0.01, b2_iv=(-0.03, 0.05))
    boot = BootstrapResult(2000, (float("-inf"), float("inf")), None, 0.3, None)
    v = verdict_p1(fit, boot, _cells(FIGURE_B, n=400), PREREG)
    c3 = next(c for c in v.conditions if c.name == "C3_drop")
    assert v.peak is None and c3.passed is False and c3.value["column"] == 2
    assert v.drop_points == pytest.approx(2.0) and v.verdict == Verdict.restricted


def test_residual_branch_is_named_below_resolution() -> None:
    fit = _fit(b1=-0.2, b2=-0.01, b2_iv=(-0.07, 0.05))  # peak far below 2, column 2
    boot = BootstrapResult(2000, (float("-inf"), 3.9), None, 0.6, None)
    table = [[80, 70, 60, 50, 40], *FIGURE_B[1:]]  # a clear decline from 2 to 6 tools
    v = verdict_p1(fit, boot, _cells(table), PREREG)
    assert v.verdict == Verdict.below_resolution
    assert any(c.name == "residual_branch" for c in v.conditions)


def test_simulated_band_survives_and_continent_does_not() -> None:
    from islands_harness.stats.fit import cluster_bootstrap, fit_p1

    for preset in ("band", "continent"):
        rows = simulate.simulate_runs(simulate.PRESETS[preset], 100, 1, f"one-{preset}")
        fit = fit_p1(rows, PREREG)
        boot = cluster_bootstrap(
            rows, PREREG, 400, snapshot_id="t", model_id="m", out_path=None, fit=fit
        )
        v = verdict_p1(fit, boot, cell_stats(cells_table(rows)), PREREG)
        if preset == "band":
            assert v.verdict == Verdict.survives and 3.0 < (v.peak or 0) < 4.0
        else:
            assert v.verdict != Verdict.survives


def _recovery_rows() -> list[dict]:
    """Two documents: each has a fault-0 control, a recovered faulted run and an unrecovered
    one in each of two tool columns."""
    rows = []
    for d, extra in (("inv-0001", 2), ("inv-0002", 4)):
        for tools in (2, 3):
            base = {"mix": "A", "tools": tools, "doc_id": d, "epoch": 0, "phase": "sweep"}
            control = {"fault_rate": 0.0, "success": True, "turns": 3, "seconds": 1.0}
            rows.append({**base, **control, "fault_events": [], "recovered": None})
            faulted = {"fault_rate": 0.2, "success": True, "turns": 3 + extra}
            rows.append(
                {
                    **base,
                    **faulted,
                    "seconds": 1.0 + extra,
                    "fault_events": [{"kind": "timeout"}],
                    "recovered": True,
                }
            )
            lost = {"fault_rate": 0.4, "success": False, "turns": 2, "seconds": 1.0}
            rows.append(
                {**base, **lost, "fault_events": [{"kind": "error_response"}], "recovered": False}
            )
    return rows


def test_recovery_pairs_bootstrap_and_counts(tmp_path: Path) -> None:
    from islands_harness.stats.recovery import analyze_recovery

    a = analyze_recovery(
        _recovery_rows(),
        PREREG,
        snapshot_id="s",
        model_id="m",
        pairing="exact_invariant_stack",
        out_dir=str(tmp_path),
    )
    assert a.n_pairs == 4 and a.n_documents == 2 and a.r == pytest.approx(3.0)
    assert 2.0 <= a.lo <= a.r <= a.hi <= 4.0
    assert a.net_of_retries == pytest.approx(2.0)  # one fault per faulted run
    assert (a.n_faulted, a.n_unrecovered, a.n_without_control) == (8, 4, 0)
    assert a.by_kind == {"timeout": 3.0} and "recovered" in a.survivorship_note
    assert (tmp_path / "resamples_p2.json").is_file()
    assert verdict_p2(a, PREREG).verdict == Verdict.survives


def test_p2_without_pairs_is_not_estimable() -> None:
    from islands_harness.stats.recovery import analyze_recovery

    rows = [r for r in _recovery_rows() if r["fault_rate"] != 0.0]  # no controls
    a = analyze_recovery(rows, PREREG, snapshot_id="s", model_id="m", pairing="x", out_dir=None)
    v = verdict_p2(a, PREREG)
    assert a.n_pairs == 0 and a.n_without_control == 4 and v.verdict == Verdict.not_estimable
    p1 = P1Verdict(Verdict.survives, [], 3.4, (2.9, 4.1), 41.0, (27.0, 53.0), "mle")
    assert statement("2026-11-15", "s1", "m", "A", p1, v).endswith("| P2: not estimable")


def _shape_cells(rates: list[float], n: int = 400) -> list[CellStat]:
    return [
        CellStat(3, f, n, round(p * n), round(p * n) / n, 0.0, 1.0, True, None, None)
        for f, p in zip(FAULTS, rates, strict=True)
    ]


def test_shape_fit_labels_a_cliff_and_a_slope() -> None:
    from islands_harness.stats.fit import shape_fit

    cliff = shape_fit(_shape_cells([0.9, 0.9, 0.89, 0.2, 0.05, 0.04]))
    slope = shape_fit(_shape_cells([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
    assert [s.column for s in cliff] == [3, None]
    assert cliff[0].label == "cliff" and 0.2 < cliff[0].f0 < 0.3
    assert slope[0].label in ("slope", "indistinguishable")
    assert slope[0].lin_slope == pytest.approx(-1.0, abs=0.05)


def test_boundary_width_on_a_steep_row() -> None:
    from islands_harness.stats.fit import boundary_width

    table = [[95, 93, 60, 20, 5], *FIGURE_A[1:]]
    rows = boundary_width(_cells(table), 0.05)["rows"]
    assert rows[0]["first_below_90"] == 4 and rows[0]["first_below_10"] == 6
    assert rows[0]["width"] == 2 and rows[0]["claimed"] is True
    assert rows[5]["width"] is None  # the 50 percent row never falls below 10 percent


def test_sensitivity_reports_both_taus_and_the_fractional_logit() -> None:
    from islands_harness.stats.fit import sensitivity

    rows = simulate.simulate_runs(simulate.PRESETS["band"], 40, 1, "sens")
    out = sensitivity(rows, PREREG, draws=200, snapshot_id="s", model_id="m")
    assert [t["tau"] for t in out["tau"]] == [0.7, 0.9] and out["descriptive"] is True
    assert out["fractional_logit"]["status"] == "fitted"


def test_jsonable_and_verdict_file_are_strict_json(tmp_path: Path) -> None:
    import json

    from islands_harness.stats.verdict import jsonable, write_verdict_json

    assert jsonable({"a": (1.0, float("inf")), "b": float("nan")}) == {
        "a": [1.0, "inf"],
        "b": None,
    }
    inf = float("inf")
    p1 = P1Verdict(Verdict.restricted, [], None, (-inf, inf), 2.0, (-9.0, 13.0), "mle")
    p2 = P2Verdict(Verdict.not_estimable, [], float("nan"), (float("nan"), float("nan")))
    line = write_verdict_json(
        tmp_path / "v.json",
        date="2026-11-15",
        snapshot="s1",
        model="m",
        mix="A",
        instrument=instrument_check(_cells(FIGURE_A), PREREG),
        p1=p1,
        p2=p2,
        prereg_hash="abc",
    )
    data = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
    assert data["p1"]["peak_interval"] == ["-inf", "inf"] and data["p2"]["r"] is None
    assert "P1: restricted (peak undefined; drop 2 points [-9, 13])" in line


def test_analyze_command_end_to_end(tmp_path: Path) -> None:
    """islands analyze on a simulated full sweep writes every file, and a second run over
    the same inputs produces byte-identical outputs."""
    import json

    from islands_harness.cli import main

    rows = simulate.simulate_runs(simulate.PRESETS["band"], 30, 1, "e2e")
    for r in rows:
        r["model_id"] = "gpt-oss-20b-local"
    out = tmp_path / "gpt-oss-20b-local"
    out.mkdir()
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (out / "runs.jsonl").write_text(lines, encoding="utf-8")
    config = str(REPO / "configs" / "snapshot-1.yaml")
    names = (
        "cells.csv",
        "results.json",
        "verdict.json",
        "statement.txt",
        "bootstrap/resamples.json",
    )
    assert main(["analyze", str(out), "--exploratory", "-c", config]) == 0
    first = {n: (out / n).read_bytes() for n in names}
    results = json.loads(first["results.json"])
    assert results["verdicts"]["p1"]["verdict"] == "survives"
    assert results["fit"]["status"] == "mle" and results["counts"]["rung1"] == 180
    resamples = json.loads(first["bootstrap/resamples.json"])
    assert resamples["statistics"]["p1"]["draws"] == 2000
    assert main(["analyze", str(out), "--exploratory", "-c", config]) == 0
    assert {n: (out / n).read_bytes() for n in names} == first


# -- fixes from the independent review of M3 -------------------------------------------------


def test_percentile_ends_open_symmetrically() -> None:
    """An end opens when more than 2.5 percent of 2,000 draws (51 or more) are infinite on
    that side, the same count on both sides."""
    import numpy as np

    from islands_harness.stats.resample import percentile_interval

    body = np.full(2000, 3.5)
    for n_inf, opens in ((50, False), (51, True)):
        low = body.copy()
        low[:n_inf] = -np.inf
        high = body.copy()
        high[:n_inf] = np.inf
        assert (percentile_interval(low, 0.05)[0] == -np.inf) is opens
        assert (percentile_interval(high, 0.05)[1] == np.inf) is opens


def test_cells_table_refuses_mixed_mixes_and_analyze_keeps_one_mix(tmp_path: Path) -> None:
    from islands_harness.stats.analyze import analyze

    band = simulate.simulate_runs(simulate.PRESETS["band"], 30, 1, "mixA", mix="A")
    other = simulate.simulate_runs(simulate.PRESETS["continent"], 30, 1, "mixB", mix="B")
    for r in other:
        r["run_id"], r["model_id"] = r["run_id"] + "-B", "sim-band"
    with pytest.raises(ValueError, match="mixes"):
        cells_table(band + other)
    common = {"snapshot_id": "s", "model_id": "sim-band", "pairing": "x", "prereg_hash": "h"}
    both = analyze(band + other, PREREG, out_dir=tmp_path / "both", **common)
    alone = analyze(band, PREREG, out_dir=tmp_path / "alone", **common)
    assert both["cells"] == alone["cells"] and len(alone["cells"]) == 36
    assert both["counts"]["runs_analyzed"] == len(band) == 1080


def test_one_document_is_not_estimable() -> None:
    from islands_harness.stats.fit import fit_p1

    rows = [
        r
        for r in simulate.simulate_runs(simulate.PRESETS["band"], 5, 1, "one")
        if r["doc_id"] == "inv-0001"
    ]
    fit = fit_p1(rows, PREREG)
    assert fit.status == "not_estimable" and "one document" in (fit.reason or "")


def test_steep_fault_slope_does_not_open_the_peak_interval() -> None:
    """A fault slope near the coefficient cap sends some resamples past 15. They are refitted
    with Firth, as the primary fit would be, instead of counting as failed; the peak interval
    stays finite and the verdict does not jump with the slope."""
    from islands_harness.stats.fit import cluster_bootstrap, fit_p1

    for b3 in (-14.0, -16.0):
        truth = simulate.Truth("steep", b0=1.6, b1=-0.6, b2=-0.6, b3=b3, document_sd=0.8)
        rows = simulate.simulate_runs(truth, 100, 1, f"steep{b3}")
        fit = fit_p1(rows, PREREG)
        boot = cluster_bootstrap(
            rows, PREREG, 400, snapshot_id="s", model_id="m", out_path=None, fit=fit
        )
        assert boot.peak_interval is not None and all(map(math.isfinite, boot.peak_interval))
        assert boot.failed_fraction == 0.0
        v = verdict_p1(fit, boot, cell_stats(cells_table(rows)), PREREG)
        assert v.verdict == Verdict.survives, (b3, fit.status, boot.firth_fraction)


def test_instrument_failure_marks_stored_verdicts_as_not_moving(tmp_path: Path) -> None:
    import json

    from islands_harness.stats.verdict import write_verdict_json

    low = [CellStat(2, 0.0, 100, 5, 0.05, *wilson(5, 100), True, None, None)]
    inst = instrument_check(low, PREREG)
    p1 = P1Verdict(Verdict.survives, [], 3.4, (2.9, 4.1), 41.0, (27.0, 53.0), "mle")
    p2 = P2Verdict(Verdict.survives, [], 1.2, (0.4, 2.0))
    write_verdict_json(
        tmp_path / "v.json",
        date="d",
        snapshot="s",
        model="m",
        mix="A",
        instrument=inst,
        p1=p1,
        p2=p2,
        prereg_hash="h",
    )
    data = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
    assert data["propositions_move"] is False
    assert data["p1"]["moves"] is False and data["p2"]["moves"] is False


def test_p2_interval_below_zero_is_restricted() -> None:
    assert verdict_p2(_recovery(-1.0, -1.6, -0.4), PREREG).verdict == Verdict.restricted


def test_read_runs_tolerates_only_a_torn_last_line(tmp_path: Path) -> None:
    from islands_harness.stats.analyze import read_runs

    good = tmp_path / "a.jsonl"
    good.write_text('{"run_id": "a"}\n{"run_id": "b"}\n{"run_id": "c', encoding="utf-8")
    rows, torn = read_runs(good)
    assert [r["run_id"] for r in rows] == ["a", "b"] and torn == 1
    bad = tmp_path / "b.jsonl"
    bad.write_text('{"run_id": "a"}\n{"run_id": \n{"run_id": "c"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_runs(bad)


def test_missing_score_and_fault_events_do_not_crash() -> None:
    from islands_harness.stats.fit import sensitivity

    rows = simulate.simulate_runs(simulate.PRESETS["band"], 20, 1, "gaps")
    rows[5]["score"] = None
    rows[6].pop("score")
    rows[7]["fault_events"] = None
    assert cells_table(rows)["n"].sum() == len(rows)
    out = sensitivity(rows, PREREG, draws=100, snapshot_id="s", model_id="m")
    assert out["fractional_logit"]["status"] == "fitted"


def test_drop_column_rounds_half_up() -> None:
    from islands_harness.stats.fit import drop_column

    assert [drop_column(t, PREREG) for t in (1.2, 2.5, 3.49, 4.5, 5.5, 9.0, None)] == [
        2,
        3,
        3,
        5,
        6,
        6,
        2,
    ]


def test_precedence_is_declared_in_the_preregistration() -> None:
    from islands_harness.stats.verdict import P1_PRECEDENCE

    assert list(PREREG.p1.precedence) == P1_PRECEDENCE
    changed = PREREG.model_copy(
        update={"p1": PREREG.p1.model_copy(update={"precedence": list(reversed(P1_PRECEDENCE))})}
    )
    fit = _fit(b1=-0.7, b2=-0.35, b2_iv=(-0.5, -0.2))
    with pytest.raises(ValueError, match="precedence"):
        verdict_p1(
            fit, BootstrapResult(2000, (2.5, 3.6), None, 1.0, None), _cells(FIGURE_A), changed
        )


def test_anchor_study_ceiling_hides_edges_and_low_anchors_do_not() -> None:
    """Reduced anchor study (method note 6): an edge under a perfect easy corner is nearly
    invisible, one under a 12 percent corner is found, and a flat ceiling never survives."""
    saturated = simulate.coverage_experiment("band_saturated", 20, 100)
    low = simulate.coverage_experiment("band_10", 20, 100)
    flat_ceiling = simulate.coverage_experiment("continent_ceiling", 20, 100)
    assert saturated.power is not None and saturated.power <= 0.2
    assert low.power == 1.0
    assert flat_ceiling.false_survival == 0.0
