"""The descriptive addendum: what the method note (Section 7) committed to report on
2026-09-27, before any sweep result was looked at. Nothing here moves a verdict; every number
sits beside the pre-registered results of r4.

``write_addendum`` writes ``addendum.json`` into the model's results directory:

    success_meaning   per cell and mix: the registered success rate (score >= tau), the rate
                      at score >= 0.9, the total-correct rate and the fully-correct rate,
                      with Wilson intervals; a run whose only error is the total scores
                      exactly 0.8 and counts as a success
    tool_effects      per mix, the rate at k tools minus the rate at k - 1, per fault rate and
                      pooled over fault rates, with a Newcombe interval (conservative for the
                      paired design), so an effect of one tool is not read as tool count
    p1_drop           the registered drop from the peak to 6 tools against the 15-point margin
    p2                the registered recovery cost, net of plain retries, and per cell the
                      share of faulted runs that never recovered
    aborts            per cell, aborted runs by cause (``abort_cause`` events)

and two descriptive re-analyses through the same ``analyze``, each labelled so that neither
can be read as a verdict:

    mix-B/                               mix B, descriptive by r4
    sensitivity/unparseable-as-failure/  mix A with runs the server rejected as unparseable
                                         model output counted as failures (score 0)
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from islands_harness.config import PreRegistration
from islands_harness.stats.fit import newcombe, wilson

ADDENDUM_FILE = "addendum.json"
MIX_B_DIR = "mix-B"
UNPARSEABLE_DIR = "sensitivity/unparseable-as-failure"
UNPARSEABLE = "unparseable_model_output"
SCHEMA = "addendum.v1"
MIX_B_READING = "mix B, descriptive by r4"
UNPARSEABLE_READING = "unparseable model output counted as failures"
SUCCESS_WORDS = (
    "A run succeeds when its weighted field score is at least tau (0.8). The weights are "
    "invoice number 2, date 1, vendor 1, currency 1, total 2 and line items 3, summing to 10, "
    "so a run whose only error is the total scores exactly 0.8 and counts as a success. "
    "Fully correct means a score of 1.0."
)


def _in_scope(runs: list[dict[str, Any]], model_id: str, mix: str) -> list[dict[str, Any]]:
    """Graded sweep runs of one model and mix with at least 2 tools (rung 1 is excluded
    from every statistic by construction)."""
    return [
        r
        for r in runs
        if r.get("model_id", model_id) == model_id
        and r.get("mix") == mix
        and r.get("phase", "sweep") == "sweep"
        and int(r.get("tools", 0)) >= 2
    ]


def _key(tools: Any, fault: Any) -> str:
    return f"{int(tools)}:{float(fault):.1f}"


def _by_cell(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[_key(r["tools"], r["fault_rate"])].append(r)
    return dict(sorted(cells.items(), key=lambda kv: tuple(map(float, kv[0].split(":")))))


def abort_cause_of(run: dict[str, Any]) -> str | None:
    """The recorded cause of an aborted run; ``unrecorded`` for an abort from before causes
    were recorded, None for a run that did not abort."""
    if run.get("outcome") != "aborted_transport":
        return None
    for e in run.get("events") or []:
        if e.get("kind") == "abort_cause":
            return str((e.get("data") or {}).get("cause"))
    return "unrecorded"


def _rate(k: int, n: int, alpha: float) -> dict[str, Any]:
    lo, hi = wilson(k, n, alpha)
    return {"k": k, "n": n, "p": k / n if n else None, "lo": lo, "hi": hi}


def success_meaning(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, cell in _by_cell([r for r in rows if r.get("success") is not None]).items():
        n = len(cell)
        field = lambda r, f: (r.get("per_field") or {}).get(f)  # noqa: E731
        out[key] = {
            "registered_success": _rate(sum(bool(r["success"]) for r in cell), n, alpha),
            "score_at_least_0_9": _rate(
                sum((r.get("score") or 0.0) >= 0.9 for r in cell), n, alpha
            ),
            "total_correct": _rate(sum(field(r, "total_amount") == 1.0 for r in cell), n, alpha),
            "fully_correct": _rate(sum(r.get("score") == 1.0 for r in cell), n, alpha),
        }
    return out


def tool_effects(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    counts: dict[tuple[int, float], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("success") is None:
            continue
        c = counts[(int(r["tools"]), round(float(r["fault_rate"]), 6))]
        c[0] += int(bool(r["success"]))
        c[1] += 1
    tools = sorted({t for t, _ in counts})
    faults = sorted({f for _, f in counts})

    def steps(get: Callable[[int], tuple[int, int] | None]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            before, after = get(t - 1), get(t)
            if before is None or after is None or not before[1] or not after[1]:
                continue
            lo, hi = newcombe(after[0], after[1], before[0], before[1], alpha)
            out.append(
                {
                    "from": t - 1,
                    "to": t,
                    "diff": after[0] / after[1] - before[0] / before[1],
                    "lo": lo,
                    "hi": hi,
                }
            )
        return out

    def at(f: float) -> Callable[[int], tuple[int, int] | None]:
        return lambda t: tuple(counts[(t, f)]) if (t, f) in counts else None  # type: ignore[return-value]

    def pooled(t: int) -> tuple[int, int] | None:
        cells = [counts[(t, f)] for f in faults if (t, f) in counts]
        return (sum(c[0] for c in cells), sum(c[1] for c in cells)) if cells else None

    return {
        "by_fault": {f"{f:.1f}": steps(at(f)) for f in faults},
        "pooled": steps(pooled),
    }


def never_recovered(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, cell in _by_cell([r for r in rows if r.get("success") is not None]).items():
        faulted = [r for r in cell if r.get("fault_events")]
        if faulted:
            never = sum(r.get("outcome") != "submitted" for r in faulted)
            out[key] = {"faulted": len(faulted), "never": never, "share": never / len(faulted)}
    return out


def aborts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, cell in _by_cell(rows).items():
        causes: dict[str, int] = defaultdict(int)
        for r in cell:
            cause = abort_cause_of(r)
            if cause is not None:
                causes[cause] += 1
        if causes:
            out[key] = dict(sorted(causes.items()))
    return out


def unparseable_as_failure(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The runs with every unparseable-model-output abort turned into a graded failure
    (success False, score 0), and how many were turned."""
    out, n = [], 0
    for r in runs:
        if abort_cause_of(r) == UNPARSEABLE:
            r = {**r, "success": False, "score": 0.0}
            n += 1
        out.append(r)
    return out, n


def _drop_reading(p1: dict[str, Any], sesoi: float) -> dict[str, Any]:
    drop, iv = p1.get("drop_points"), p1.get("drop_interval")
    if drop is None or iv is None:
        words = "no drop is defined: the fit has no interior peak, so there is no peak to drop from"
    elif iv[0] >= sesoi:
        words = f"the drop is at least the {sesoi:g}-point margin"
    elif iv[1] < sesoi:
        words = f"the drop is smaller than the {sesoi:g}-point margin"
    else:
        words = f"the drop cannot be told apart from the {sesoi:g}-point margin"
    return {
        "verdict": p1.get("verdict"),
        "drop_points": drop,
        "drop_interval": iv,
        "sesoi_points": sesoi,
        "reading": words,
    }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statement": payload["statement"],
        "p1": payload["verdicts"]["p1"]["verdict"],
        "p2": payload["verdicts"]["p2"]["verdict"],
    }


def write_addendum(
    runs: list[dict[str, Any]],
    prereg: PreRegistration,
    registered: dict[str, Any],
    *,
    snapshot_id: str,
    model_id: str,
    pairing: str,
    prereg_hash: str,
    out_dir: Path,
    torn_lines: int = 0,
) -> dict[str, Any]:
    """Compute and write the addendum for one model; ``registered`` is the results.json
    payload of the pre-registered analysis (mix A). Returns the addendum payload."""
    from islands_harness.stats.analyze import _write_json, analyze

    out_dir = Path(out_dir)
    alpha = prereg.alpha
    mix = str(registered["mix"])
    common = {
        "snapshot_id": snapshot_id,
        "model_id": model_id,
        "pairing": pairing,
        "prereg_hash": prereg_hash,
        "torn_lines": torn_lines,
    }
    mixes = sorted({str(r.get("mix")) for r in runs if r.get("phase", "sweep") == "sweep"})
    rows = {m: _in_scope(runs, model_id, m) for m in mixes}

    turned, n_turned = unparseable_as_failure(runs)
    if n_turned:
        sensitivity = _summary(
            analyze(
                turned,
                prereg,
                out_dir=out_dir / UNPARSEABLE_DIR,
                mix=mix,
                reading=UNPARSEABLE_READING,
                **common,
            )
        )
        sensitivity["differs"] = {
            p: sensitivity[p] != registered["verdicts"][p]["verdict"] for p in ("p1", "p2")
        }
    else:
        sensitivity = {
            "note": "no run ended as unparseable model output; identical to the registered reading"
        }
    sensitivity["runs_counted_as_failures"] = n_turned

    mix_b = None
    if "B" in mixes and mix != "B" and rows.get("B"):
        mix_b = {
            **_summary(
                analyze(
                    runs,
                    prereg,
                    out_dir=out_dir / MIX_B_DIR,
                    mix="B",
                    reading=MIX_B_READING,
                    **common,
                )
            ),
            "dir": MIX_B_DIR,
        }

    recovery = registered["recovery"]
    payload = {
        "schema": SCHEMA,
        "descriptive": True,
        "decided": "2026-09-27, before any sweep result was looked at (method note Section 7)",
        "snapshot": snapshot_id,
        "model": model_id,
        "registered_mix": mix,
        "success_meaning": {
            "words": SUCCESS_WORDS,
            "by_mix": {m: success_meaning(rows[m], alpha) for m in mixes},
        },
        "tool_effects": {m: tool_effects(rows[m], alpha) for m in mixes},
        "p1_drop": _drop_reading(registered["verdicts"]["p1"], prereg.p1.drop.sesoi_points),
        "p2": {
            "verdict": registered["verdicts"]["p2"]["verdict"],
            "turns": registered["verdicts"]["p2"]["r"],
            "interval": registered["verdicts"]["p2"]["interval"],
            "net_of_retries": {
                "turns": recovery.get("net_of_retries"),
                "lo": recovery.get("net_lo"),
                "hi": recovery.get("net_hi"),
            },
            "never_recovered": {m: never_recovered(rows[m]) for m in mixes},
        },
        "aborts": {m: aborts(rows[m]) for m in mixes},
        "unparseable_as_failure": {**sensitivity, "dir": UNPARSEABLE_DIR if n_turned else None},
        "mix_b": mix_b,
    }
    _write_json(out_dir / ADDENDUM_FILE, payload)
    return payload


def read_addendum(out_dir: Path) -> dict[str, Any] | None:
    path = Path(out_dir) / ADDENDUM_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
