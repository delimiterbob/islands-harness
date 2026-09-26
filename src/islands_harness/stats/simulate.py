"""Synthetic snapshots of known truth, for coverage and false-survival tests.

The instrument is verified against truths it did not see: a band with a peak at 3.5 tools,
a continent with b2 = 0, and a flat surface below resolution, each with document
heterogeneity in pi_d (ARCHITECTURE.md Section 9). ``test_stats.py`` runs the coverage
experiment at n = 100 and n = 50 documents and asserts Wilson coverage in [93, 97] percent,
peak-interval coverage of at least 93 percent, and a false-survival rate under the
continent of at most 5 percent; power under the band is reported, not asserted. The method
note quotes these numbers from ``main()``.

Generative model, per document d and run (tools t, fault f):

    eta_d = b0 + b1 tc + b2 tc2 + b3 f + gamma_d,   gamma_d ~ N(0, document_sd) via a hashed
    normal draw (Box-Muller on two rng.unit draws); success ~ Bernoulli(sigmoid(eta_d)).

Every draw is ``rng.unit(scope, preset, snapshot_index, doc, tools, fault_pct, field)`` so
a simulated snapshot is reproducible from its index.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from islands_harness.config import PreRegistration
from islands_harness.rng import unit

DEFAULT_PREREG = Path(__file__).resolve().parents[3] / "configs" / "preregistration" / "r4.yaml"


@dataclass(frozen=True)
class Truth:
    name: str
    b0: float
    b1: float
    b2: float
    b3: float
    document_sd: float

    @property
    def peak(self) -> float | None:
        return None if self.b2 >= 0 else 4.0 - self.b1 / (2.0 * self.b2)


# Presets. The band's peak is 3.5 tools: 4 - b1 / (2 b2) = 3.5 -> b1 = b2 (with b2 < 0).
PRESETS: dict[str, Truth] = {
    "band": Truth("band", b0=1.6, b1=-0.6, b2=-0.6, b3=-3.0, document_sd=0.8),
    "continent": Truth("continent", b0=1.4, b1=0.0, b2=0.0, b3=-3.0, document_sd=0.8),
    "flat": Truth("flat", b0=0.4, b1=0.0, b2=0.0, b3=-0.3, document_sd=0.8),
}

# The anchor study (2026-09-26): the same two shapes with the easy corner moved to the
# ceiling or well below the calibration band, to measure whether the verdicts still hold
# there. Anchor rates (2 tools, no faults, averaged over document effects): band_ceiling
# 0.97 (peak 0.99, 6 tools 0.77), band_40 0.45, band_20 0.26; the continents are flat at
# 0.97, 0.40 and 0.19. band_10 has an easy corner near 0.12, and band_saturated one at
# 0.998, a perfect anchor in practice. The existing band preset's anchor is 0.58.
ANCHOR_PRESETS: dict[str, Truth] = {
    "band_ceiling": Truth("band_ceiling", b0=5.0, b1=-0.6, b2=-0.6, b3=-3.0, document_sd=0.8),
    "band_40": Truth("band_40", b0=1.0, b1=-0.6, b2=-0.6, b3=-3.0, document_sd=0.8),
    "band_20": Truth("band_20", b0=0.0, b1=-0.6, b2=-0.6, b3=-3.0, document_sd=0.8),
    "band_10": Truth("band_10", b0=-1.0, b1=-0.6, b2=-0.6, b3=-3.0, document_sd=0.8),
    "band_saturated": Truth("band_saturated", b0=8.0, b1=-0.6, b2=-0.6, b3=-3.0, document_sd=0.8),
    "continent_ceiling": Truth(
        "continent_ceiling", b0=3.8, b1=0.0, b2=0.0, b3=-3.0, document_sd=0.8
    ),
    "continent_40": Truth("continent_40", b0=-0.45, b1=0.0, b2=0.0, b3=-3.0, document_sd=0.8),
    "continent_20": Truth("continent_20", b0=-1.6, b1=0.0, b2=0.0, b3=-3.0, document_sd=0.8),
}

TOOLS_AXIS = [1, 2, 3, 4, 5, 6]
FAULT_AXIS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normal(scope: str, *parts: Any) -> float:
    """Standard normal via Box-Muller on two hashed uniforms (the first one shifted off 0)."""
    u1 = 1.0 - unit(scope, *parts, "u1")
    u2 = unit(scope, *parts, "u2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def document_effects(truth: Truth, n_docs: int, scope: str) -> list[float]:
    """gamma_d for each document of one simulated snapshot."""
    return [truth.document_sd * _normal("sim-doc", scope, truth.name, d) for d in range(n_docs)]


def linear_predictor(truth: Truth, tools: int, fault: float) -> float:
    tc = tools - 4.0
    return truth.b0 + truth.b1 * tc + truth.b2 * tc * tc + truth.b3 * fault


def true_rate(truth: Truth, tools: int, fault: float, gamma: list[float]) -> float:
    """pi_c: the mean over this snapshot's documents of sigmoid(eta_d), the Section 9
    estimand. Rung 1 is 0 by construction."""
    if tools < 2:
        return 0.0
    eta = linear_predictor(truth, tools, fault)
    return mean(_sigmoid(eta + g) for g in gamma)


def simulate_runs(
    truth: Truth, n_docs: int, epochs: int, scope: str, *, mix: str = "A", turns_base: int = 3
) -> list[dict[str, Any]]:
    """Rows shaped like runs.jsonl for a full 6 by 6 grid from a known truth.

    Fields present: run_id, model_id, phase, mix, tools, fault_rate, index, doc_id, epoch,
    outcome, success, score (= success as float), turns, seconds, fault_events (empty; the
    P2 population is not simulated here), recovered.
    """
    rows: list[dict[str, Any]] = []
    gamma = document_effects(truth, n_docs, scope)
    for tools in TOOLS_AXIS:
        for fault in FAULT_AXIS:
            fault_pct = int(round(fault * 100))
            for epoch in range(epochs):
                for d in range(n_docs):
                    p = _sigmoid(linear_predictor(truth, tools, fault) + gamma[d])
                    u = unit("sim-run", scope, truth.name, d, epoch, tools, fault_pct)
                    success = tools >= 2 and u < p
                    rows.append(
                        {
                            "run_id": f"sim-{scope}-{tools}-{fault_pct}-{epoch}-{d}",
                            "model_id": f"sim-{truth.name}",
                            "phase": "sweep",
                            "mix": mix,
                            "tools": tools,
                            "fault_rate": fault,
                            "index": epoch * n_docs + d,
                            "doc_id": f"inv-{d + 1:04d}",
                            "epoch": epoch,
                            "outcome": "submitted" if success else "no_submission",
                            "success": bool(success),
                            "score": 1.0 if success else 0.0,
                            "turns": turns_base,
                            "seconds": 1.0,
                            "fault_events": [],
                            "recovered": None,
                        }
                    )
    return rows


@dataclass(frozen=True)
class CoverageReport:
    preset: str
    n_snapshots: int
    n_docs: int
    wilson_coverage: float  # share of (snapshot, in-scope cell) whose Wilson interval covers pi_c
    peak_coverage: float | None  # share of snapshots whose t* interval covers the true peak
    false_survival: float | None  # share of snapshots where P1 survives, truths without a peak
    power: float | None  # share of snapshots where P1 survives, truths with a peak
    not_estimable_rate: float
    firth_rate: float = 0.0  # share of snapshots whose fit moved to the Firth rung
    mean_failed_fraction: float = 0.0  # mean share of bootstrap refits that failed their rung
    verdicts: dict[str, int] | None = None  # verdict counts over the snapshots
    residual_rate: float = 0.0  # share of snapshots labelled by the unnamed residual branch


def coverage_experiment(
    preset: str,
    n_snapshots: int,
    n_docs: int,
    *,
    epochs: int = 1,
    draws: int | None = None,
    prereg: PreRegistration | None = None,
) -> CoverageReport:
    """Simulate ``n_snapshots`` snapshots from ``PRESETS[preset]``, run fit_p1, the bootstrap
    and verdict_p1 on each, and tabulate coverage, false survival and power.

    The true cell rate pi_c is the average of sigmoid(eta_d) over the snapshot's documents,
    the estimand in Section 9, so Wilson coverage is checked against it and not against the
    rate without document effects. The true peak is the conditional one, 4 - b1 / (2 b2):
    with a document effect added on the logit scale, every document's curve and their
    average peak at the same tool count, because the average of sigmoid(eta + gamma) is
    increasing in eta. Bootstrap draws default to the pre-registered count; the resample
    index table is shared across snapshots of one size, which is valid because each
    snapshot's data are independent of it.
    """
    from islands_harness.config import load_prereg
    from islands_harness.stats.fit import cell_stats, cells_table, cluster_bootstrap, fit_p1
    from islands_harness.stats.verdict import Verdict, verdict_p1

    prereg = prereg if prereg is not None else load_prereg(DEFAULT_PREREG)
    draws = draws if draws is not None else prereg.p1.bootstrap.draws
    truth = PRESETS[preset] if preset in PRESETS else ANCHOR_PRESETS[preset]
    covered = cells_seen = peak_hits = firth = not_estimable = residual = 0
    failed: list[float] = []
    verdicts: dict[str, int] = {}
    for i in range(n_snapshots):
        scope = f"cov-{preset}-{n_docs}-{epochs}-{i}"
        rows = simulate_runs(truth, n_docs, epochs, scope)
        gamma = document_effects(truth, n_docs, scope)
        cells = cell_stats(cells_table(rows))
        for c in cells:
            if c.in_scope:
                cells_seen += 1
                covered += c.lo <= true_rate(truth, c.tools, c.fault, gamma) <= c.hi
        fit = fit_p1(rows, prereg)
        boot = cluster_bootstrap(
            rows,
            prereg,
            draws,
            snapshot_id=f"sim-{n_docs}",
            model_id="sim",
            out_path=None,
            fit=fit,
        )
        verdict = verdict_p1(fit, boot, cells, prereg)
        verdicts[verdict.verdict.value] = verdicts.get(verdict.verdict.value, 0) + 1
        residual += any(c.name == "residual_branch" for c in verdict.conditions)
        firth += fit.status == "firth"
        not_estimable += fit.status == "not_estimable"
        failed.append(boot.failed_fraction)
        if truth.peak is not None and boot.peak_interval is not None:
            lo, hi = boot.peak_interval
            peak_hits += lo <= truth.peak <= hi
    survived = verdicts.get(Verdict.survives.value, 0) / n_snapshots
    return CoverageReport(
        preset=preset,
        n_snapshots=n_snapshots,
        n_docs=n_docs,
        wilson_coverage=covered / cells_seen if cells_seen else float("nan"),
        peak_coverage=peak_hits / n_snapshots if truth.peak is not None else None,
        false_survival=None if truth.peak is not None else survived,
        power=survived if truth.peak is not None else None,
        not_estimable_rate=not_estimable / n_snapshots,
        firth_rate=firth / n_snapshots,
        mean_failed_fraction=mean(failed) if failed else 0.0,
        verdicts=dict(sorted(verdicts.items())),
        residual_rate=residual / n_snapshots,
    )


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def main() -> None:
    """Print the coverage table the method note quotes: every preset at n = 100 and n = 50,
    500 snapshots each unless ISLANDS_SIMULATION_SNAPSHOTS says otherwise."""
    n = int(os.environ.get("ISLANDS_SIMULATION_SNAPSHOTS", "500"))
    head = (
        "| truth | documents | snapshots | Wilson coverage % | peak coverage % | "
        "false survival % | power % | residual branch % | Firth % | failed refits % | verdicts |"
    )
    print(head)
    print("|" + "---|" * 11)
    for preset in PRESETS:
        for n_docs in (100, 50):
            r = coverage_experiment(preset, n, n_docs)
            cols = [
                r.preset,
                str(r.n_docs),
                str(r.n_snapshots),
                _pct(r.wilson_coverage),
                _pct(r.peak_coverage),
                _pct(r.false_survival),
                _pct(r.power),
                _pct(r.residual_rate),
                _pct(r.firth_rate),
                _pct(r.mean_failed_fraction),
                str(r.verdicts),
            ]
            print("| " + " | ".join(cols) + " |", flush=True)


def anchor_main() -> None:
    """Print the anchor study the method note quotes: each anchor preset at 100 documents,
    500 snapshots unless ISLANDS_SIMULATION_SNAPSHOTS says otherwise."""
    n = int(os.environ.get("ISLANDS_SIMULATION_SNAPSHOTS", "500"))
    print(
        "| truth | Wilson coverage % | peak coverage % | false survival % | power % | residual branch % | Firth % | verdicts |"
    )
    print("|" + "---|" * 8)
    for preset in ANCHOR_PRESETS:
        r = coverage_experiment(preset, n, 100)
        cols = [
            r.preset,
            _pct(r.wilson_coverage),
            _pct(r.peak_coverage),
            _pct(r.false_survival),
            _pct(r.power),
            _pct(r.residual_rate),
            _pct(r.firth_rate),
            str(r.verdicts),
        ]
        print("| " + " | ".join(cols) + " |", flush=True)


if __name__ == "__main__":  # pragma: no cover
    import sys

    anchor_main() if "--anchor" in sys.argv else main()
