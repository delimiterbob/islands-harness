"""The descriptive addendum (method note Section 7): its per-cell numbers on hand-made runs,
and a whole analysis with mix B and the unparseable-output reading on the synthetic agent,
checked by the reviewer's re-analysis."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from islands_harness.config import load_config, load_prereg, prereg_hash
from islands_harness.stats.addendum import (
    UNPARSEABLE,
    abort_cause_of,
    aborts,
    never_recovered,
    success_meaning,
    tool_effects,
    unparseable_as_failure,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "snapshot-1.yaml"


def _run(tools: int, fault: float, *, score: float | None, total: float = 0.0, **extra) -> dict:  # noqa: ANN003
    success = None if score is None else score >= 0.8
    return {
        "tools": tools,
        "fault_rate": fault,
        "mix": "A",
        "phase": "sweep",
        "score": score,
        "success": success,
        "per_field": None if score is None else {"total_amount": total},
        "outcome": extra.pop("outcome", "submitted" if score else "no_submission"),
        "fault_events": extra.pop("fault_events", []),
        "events": extra.pop("events", []),
        **extra,
    }


def test_the_addendum_numbers_on_hand_made_runs() -> None:
    rows = [
        _run(2, 0.0, score=1.0, total=1.0),
        _run(2, 0.0, score=0.8),  # everything right but the total: a registered success
        _run(2, 0.0, score=0.5),
        _run(3, 0.0, score=1.0, total=1.0),
        _run(3, 0.0, score=1.0, total=1.0),
        _run(3, 0.0, score=0.0, fault_events=[{"kind": "timeout"}], outcome="no_submission"),
        _run(3, 0.0, score=0.9, total=1.0, fault_events=[{"kind": "timeout"}]),
    ]
    cells = success_meaning(rows, 0.05)
    two = cells["2:0.0"]
    assert (
        two["registered_success"]["k"],
        two["fully_correct"]["k"],
        two["total_correct"]["k"],
    ) == (2, 1, 1)
    assert two["score_at_least_0_9"]["k"] == 1 and two["registered_success"]["n"] == 3
    effect = tool_effects(rows, 0.05)["pooled"]
    assert [(e["from"], e["to"]) for e in effect] == [(2, 3)]
    assert (
        abs(effect[0]["diff"] - (3 / 4 - 2 / 3)) < 1e-12
        and effect[0]["lo"] < effect[0]["diff"] < effect[0]["hi"]
    )
    assert never_recovered(rows)["3:0.0"] == {"faulted": 2, "never": 1, "share": 0.5}

    aborted = _run(
        4,
        0.3,
        score=None,
        outcome="aborted_transport",
        events=[{"kind": "abort_cause", "turn": 0, "data": {"cause": UNPARSEABLE}}],
    )
    old = _run(4, 0.3, score=None, outcome="aborted_transport")
    assert abort_cause_of(aborted) == UNPARSEABLE and abort_cause_of(old) == "unrecorded"
    assert abort_cause_of(rows[0]) is None
    assert aborts([aborted, old]) == {"4:0.3": {UNPARSEABLE: 1, "unrecorded": 1}}
    turned, n = unparseable_as_failure([aborted, old, rows[0]])
    assert n == 1 and (turned[0]["success"], turned[0]["score"]) == (False, 0.0)
    assert turned[1]["success"] is None  # only the model's own unparseable output is turned


def _synthetic_sweep(out: Path) -> tuple:  # noqa: ANN202
    from islands_harness.provenance import selftest_model
    from islands_harness.providers.synthetic import SyntheticAgent
    from islands_harness.runner import Storage, load_task
    from islands_harness.runner import run_specs as drive
    from islands_harness.specs import expand_cells, run_specs

    cfg = load_config(CONFIG)
    model = selftest_model(cfg).model_copy(update={"documents": 8})
    task = load_task(cfg, REPO)
    agent = SyntheticAgent(
        {d: g.fields for d, g in task.gold.items()}, task.documents, model_id=model.id
    )
    specs = [
        s
        for mix in ("A", "B")
        for cell in expand_cells(cfg, mix)
        for s in run_specs(cfg, model, cell, sorted(task.documents))
    ]
    prereg = load_prereg(REPO / cfg.snapshot.preregistration)
    asyncio.run(
        drive(
            specs,
            agent,
            task,
            Storage(out),
            None,
            config=cfg,
            model=model,
            tau=prereg.success.tau,
            concurrency=1,
        )
    )
    return cfg, model, prereg


def _analyse(out: Path, model, prereg) -> tuple[dict, dict]:  # noqa: ANN001
    from islands_harness.stats.addendum import write_addendum
    from islands_harness.stats.analyze import analyze, read_runs

    runs, torn = read_runs(out / "runs.jsonl")
    common = {
        "snapshot_id": "s1-test",
        "model_id": model.id,
        "pairing": "exact_synthetic_agent",
        "prereg_hash": prereg_hash(prereg),
        "torn_lines": torn,
    }
    payload = analyze(runs, prereg, out_dir=out, **common)
    return payload, write_addendum(runs, prereg, payload, out_dir=out, **common)


def test_addendum_with_mix_b_and_the_unparseable_reading(tmp_path: Path) -> None:
    from islands_harness.verification import check_analysis

    out = tmp_path / "model"
    cfg, model, prereg = _synthetic_sweep(out)
    registered, add = _analyse(out, model, prereg)

    assert add["descriptive"] is True and set(add["success_meaning"]["by_mix"]) == {"A", "B"}
    for cell in add["success_meaning"]["by_mix"]["A"].values():
        assert cell["fully_correct"]["k"] <= cell["registered_success"]["k"]
    assert [(e["from"], e["to"]) for e in add["tool_effects"]["A"]["pooled"]] == [
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ]
    assert add["p1_drop"]["verdict"] == registered["verdicts"]["p1"]["verdict"]
    assert add["mix_b"]["statement"].startswith("DESCRIPTIVE (mix B, descriptive by r4) | ")
    mix_b = json.loads((out / "mix-B" / "results.json").read_text(encoding="utf-8"))
    assert mix_b["mix"] == "B" and mix_b["propositions_move"] is False
    assert mix_b["counts"]["in_scope_p1"] > 0  # the B reading sees mix B's runs
    assert add["mix_b"]["p1"] != "not_estimable" and add["mix_b"]["p2"] != "not_estimable"
    assert (
        json.loads((out / "mix-B" / "verdict.json").read_text(encoding="utf-8"))["moves"] is False
    )
    assert add["unparseable_as_failure"]["runs_counted_as_failures"] == 0
    assert not (out / "sensitivity").exists()
    assert "DESCRIPTIVE" not in (out / "statement.txt").read_text(encoding="utf-8")
    assert check_analysis(out, config=cfg, repo_root=REPO).ok

    # One registered run turns into an unparseable-output abort: the reading appears.
    lines = (out / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        row = json.loads(line)
        if row["mix"] == "A" and row["tools"] == 3 and row["fault_rate"] == 0.0 and row["success"]:
            row.update(
                outcome="aborted_transport",
                success=None,
                score=None,
                per_field=None,
                events=[
                    *row["events"],
                    {"kind": "abort_cause", "turn": 1, "data": {"cause": UNPARSEABLE}},
                ],
            )
            lines[i] = json.dumps(row, sort_keys=True)
            break
    (out / "runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    registered, add = _analyse(out, model, prereg)
    assert registered["counts"]["aborted_transport"] == 1
    reading = add["unparseable_as_failure"]
    assert (
        reading["runs_counted_as_failures"] == 1
        and reading["dir"] == "sensitivity/unparseable-as-failure"
    )
    assert reading["statement"].startswith(
        "DESCRIPTIVE (unparseable model output counted as failures) | "
    )
    assert add["aborts"]["A"] == {"3:0.0": {UNPARSEABLE: 1}}
    assert check_analysis(out, config=cfg, repo_root=REPO).ok
