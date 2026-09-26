"""One model's analysis: runs.jsonl rows and the frozen pre-registration in, every statistic,
the verdicts and the statement out (ARCHITECTURE.md Sections 9, 11 and 12).

``analyze`` is a pure function of its inputs: nothing reads the clock, rows are sorted by
run id before any arithmetic (so the order in which concurrent runs finished cannot move a
floating-point sum), and the statement's date is the last request date found in the runs.
Re-running it on the same inputs and the same software stack gives byte-identical files,
which is what ``analyze --check`` will diff in M4. Everything is computed before anything is
written, so a failure cannot leave new verdict files beside a stale results.json; a
descriptive statistic that fails records its error in results.json instead of stopping the
analysis. Files written to the results directory:

    cells.csv                  per-cell counts, rates, Wilson limits and faults per run
    results.json               everything below, with ``descriptive: true`` on descriptive blocks
    bootstrap/resamples.json   the resample indices of every bootstrap statistic (p1, p2, p2_net)
    verdict.json               every condition with its value and pass flag
    statement.txt              the one-line dated statement
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from islands_harness.config import PreRegistration
from islands_harness.stats.fit import (
    boundary_width,
    cell_stats,
    cells_table,
    cluster_bootstrap,
    design_matrix,
    fit_p1,
    sensitivity,
    shape_fit,
    viable_fraction,
)
from islands_harness.stats.recovery import analyze_recovery, pair_controls
from islands_harness.stats.resample import resample_indices, resample_record, write_resamples
from islands_harness.stats.verdict import (
    instrument_check,
    jsonable,
    verdict_p1,
    verdict_p2,
    write_verdict_json,
)

SCHEMA_VERSION = 1


def read_runs(path: Path) -> tuple[list[dict[str, Any]], int]:
    """runs.jsonl rows and the number of torn lines skipped. Only a final line without its
    newline (a crash mid-write, which resume also truncates) may be torn; a malformed line
    anywhere else means the file is damaged, and reading stops with an error."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    rows: list[dict[str, Any]] = []
    torn = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:  # the last piece has no newline after it
                torn += 1
                continue
            raise ValueError(f"{path}: line {i + 1} is not valid JSON: {exc}") from exc
    return rows, torn


def last_request_date(runs: list[dict[str, Any]]) -> str:
    """YYYY-MM-DD of the latest model request recorded in the runs, or "undated"."""
    dates = [
        str(e["data"]["identity"]["request_date"])[:10]
        for r in runs
        for e in r.get("events") or []
        if e.get("kind") == "model_turn"
        and ((e.get("data") or {}).get("identity") or {}).get("request_date")
    ]
    return max(dates) if dates else "undated"


def malformed_rates(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Malformed tool calls over all tool calls, per cell "tools:fault". Descriptive."""
    calls: dict[str, list[int]] = {}
    for r in runs:
        if r.get("success") is None:
            continue
        key = f"{int(r['tools'])}:{float(r['fault_rate']):.1f}"
        c = calls.setdefault(key, [0, 0])
        c[0] += int(r.get("malformed_calls") or 0)
        c[1] += int(r.get("tool_calls") or 0)
    return {k: (m / t if t else 0.0) for k, (m, t) in sorted(calls.items())}


def _descriptive(compute: Callable[[], Any]) -> Any:
    """Run one descriptive statistic. A failure is recorded, not raised: descriptive numbers
    never decide anything, so they must never stop the decision from being written."""
    try:
        return compute()
    except Exception as exc:  # noqa: BLE001 - recorded in results.json, see docstring
        return {"descriptive": True, "error": f"{type(exc).__name__}: {exc}"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def analyze(
    runs: list[dict[str, Any]],
    prereg: PreRegistration,
    *,
    snapshot_id: str,
    model_id: str,
    pairing: str,
    prereg_hash: str,
    out_dir: Path,
    mix: str | None = None,
    torn_lines: int = 0,
) -> dict[str, Any]:
    """Run every P1 and P2 statistic for one model and one mix, then write the files listed
    above. Only sweep-phase rows of that model and mix enter. Returns the results.json
    payload."""
    out_dir = Path(out_dir)
    mix = mix or prereg.p1.scope.mix
    rows = sorted(
        (
            r
            for r in runs
            if r.get("model_id", model_id) == model_id
            and r.get("mix") == mix
            and r.get("phase", "sweep") == "sweep"
        ),
        key=lambda r: str(r.get("run_id", "")),
    )
    date = last_request_date(rows)

    # -- compute ----------------------------------------------------------------------------
    table = cells_table(rows)
    cells = cell_stats(table)
    fit = fit_p1(rows, prereg)
    draws = prereg.p1.bootstrap.draws
    boot = cluster_bootstrap(
        rows, prereg, draws, snapshot_id=snapshot_id, model_id=model_id, out_path=None, fit=fit
    )
    recovery = analyze_recovery(
        rows, prereg, snapshot_id=snapshot_id, model_id=model_id, pairing=pairing, out_dir=None
    )
    instrument = instrument_check(cells, prereg)
    p1 = verdict_p1(fit, boot, cells, prereg)
    p2 = verdict_p2(recovery, prereg)

    statistics: dict[str, Any] = {}  # every bootstrap's indices, from the same cached tables
    design = design_matrix(rows, prereg)
    if fit.status != "not_estimable" and not design.empty:
        docs = sorted(str(d) for d in design["doc_id"].unique())
        statistics["p1"] = resample_record(
            docs, resample_indices(snapshot_id, model_id, "p1", draws, len(docs))
        )
    pairs = pair_controls(rows, prereg)
    if pairs:
        docs = sorted({p.doc_id for p in pairs})
        for name in ("p2", "p2_net"):
            idx = resample_indices(
                snapshot_id, model_id, name, prereg.p2.bootstrap.draws, len(docs)
            )
            statistics[name] = resample_record(docs, idx)

    descriptive = {
        "viable_fraction": _descriptive(lambda: {**viable_fraction(cells), "descriptive": True}),
        "boundary_width": _descriptive(lambda: boundary_width(cells, prereg.alpha)),
        "fault_axis_shape": _descriptive(
            lambda: {"descriptive": True, "fits": [asdict(s) for s in shape_fit(cells)]}
        ),
        "sensitivity": _descriptive(
            lambda: sensitivity(
                rows, prereg, draws=draws, snapshot_id=snapshot_id, model_id=model_id
            )
        ),
        "malformed_call_rate": _descriptive(
            lambda: {"descriptive": True, "by_cell": malformed_rates(rows)}
        ),
        "recovery_extras": {
            "descriptive": True,
            "net_of_retries": [recovery.net_of_retries, recovery.net_lo, recovery.net_hi],
            "seconds_mean": recovery.seconds_mean,
            "by_kind": recovery.by_kind,
            "by_row": recovery.by_row,
        },
    }
    counts = {
        "runs_in_file": len(runs),
        "runs_analyzed": len(rows),
        "torn_lines_skipped": torn_lines,
        "aborted_transport": sum(1 for r in rows if r.get("success") is None),
        "rung1": sum(1 for r in rows if int(r.get("tools", 0)) == 1),
        "in_scope_p1": int(len(design)),
        "documents_p1": int(design["doc_id"].nunique()) if not design.empty else 0,
    }

    # -- write --------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "cells.csv", index=False, lineterminator="\n", float_format="%.10g")
    write_resamples(
        out_dir / "bootstrap" / "resamples.json",
        snapshot_id=snapshot_id,
        model_id=model_id,
        statistics=statistics,
    )
    line = write_verdict_json(
        out_dir / "verdict.json",
        date=date,
        snapshot=snapshot_id,
        model=model_id,
        mix=mix,
        instrument=instrument,
        p1=p1,
        p2=p2,
        prereg_hash=prereg_hash,
    )
    (out_dir / "statement.txt").write_text(line + "\n", encoding="utf-8", newline="\n")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot": snapshot_id,
        "model": model_id,
        "mix": mix,
        "date": date,
        "preregistration": {"version": prereg.version, "hash": prereg_hash},
        "counts": counts,
        "cells": [asdict(c) for c in cells],
        "fit": asdict(fit),
        "bootstrap": {**asdict(boot), "resamples_path": "bootstrap/resamples.json"},
        "recovery": asdict(recovery),
        "instrument": asdict(instrument),
        "propositions_move": instrument.ok,
        "verdicts": {
            "p1": {**asdict(p1), "verdict": p1.verdict.value, "moves": instrument.ok},
            "p2": {**asdict(p2), "verdict": p2.verdict.value, "moves": instrument.ok},
        },
        "statement": line,
        "descriptive": descriptive,
    }
    _write_json(out_dir / "results.json", payload)
    return payload
