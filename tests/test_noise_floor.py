"""The noise floor (M7): its specs, its statistics on hand-made runs, and a whole floor on
the synthetic agent through the real loop, task tools, runner and resume."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from islands_harness.config import SnapshotConfig, load_config
from islands_harness.specs import Cell, noise_floor_specs, run_id

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "snapshot-1.yaml"
DOCS = [f"inv-{i:04d}" for i in range(1, 101)]


def _small(cfg: SnapshotConfig, *, fixed: int, block: int, varied: int) -> SnapshotConfig:
    local = cfg.noise_floor.local.model_copy(
        update={
            "fixed_seed_reruns": fixed,
            "fixed_seed_documents": block,
            "varied_seed_reruns": varied,
        }
    )
    return cfg.model_copy(
        update={"noise_floor": cfg.noise_floor.model_copy(update={"local": local})}
    )


def test_floor_specs_repeat_the_block_and_vary_only_the_seed() -> None:
    cfg = load_config(CONFIG)
    plan = noise_floor_specs(cfg, cfg.model_by_id("qwen3-4b-local"), DOCS, "A")
    fixed, varied = plan["fixed"], plan["varied"]
    assert (len(fixed), len(fixed[0])) == (50, 20)
    assert (len(varied), len(varied[0])) == (50, 100)
    assert all(s.cell == Cell(3, 0.2, "A") for rerun in fixed + varied for s in rerun)
    for i in range(20):  # identical specs: only the stack can make fixed reruns differ
        assert (
            len({(r[i].doc_id, r[i].sample_seed, r[i].epoch, r[i].tool_list) for r in fixed}) == 1
        )
    for i in (0, 57, 99):  # same document, a new seed in every varied rerun
        assert len({r[i].doc_id for r in varied}) == 1
        assert len({r[i].sample_seed for r in varied}) == 50
    assert [r[0].rerun for r in fixed] == list(range(50))
    assert [r[0].seed_offset for r in varied] == [r[0].rerun for r in varied] == list(range(50))
    assert {s.phase for r in fixed for s in r} == {"floor_fixed"}
    ids = [s.run_id for rerun in fixed + varied for s in rerun]
    assert len(ids) == len(set(ids)) == 50 * 20 + 50 * 100
    hosted = plan["hosted"]
    assert len(hosted) == cfg.noise_floor.hosted.reruns
    assert (
        len(hosted[0])
        == len(cfg.noise_floor.hosted.cells) * cfg.noise_floor.hosted.documents_per_rerun
    )


def test_run_ids_outside_the_floor_are_unchanged_by_the_rerun_field() -> None:
    ident = {
        "model_id": "m",
        "mix": "A",
        "tools": 2,
        "fault_rate": 0.0,
        "index": 7,
        "phase": "gate",
        "seed_offset": None,
    }
    before = hashlib.sha256(
        json.dumps(ident, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    assert run_id("m", Cell(2, 0.0, "A"), 7, "gate", None) == before
    assert run_id("m", Cell(2, 0.0, "A"), 7, "gate", None, rerun=0) != before


def _store(floor_dir: Path, specs: list, success: list, texts: list[str]) -> None:  # noqa: ANN001
    """runs.jsonl rows and one-message transcripts for hand-made runs."""
    from islands_harness.providers.base import Message
    from islands_harness.runner import Storage

    storage = Storage(floor_dir)
    with storage.runs_path.open("a", encoding="utf-8", newline="\n") as f:
        for spec, ok, text in zip(specs, success, texts, strict=True):
            path = storage.write_transcript(spec.run_id, [Message("assistant", text)])
            row = {
                "run_id": spec.run_id,
                "success": ok,
                "transcript_path": path.relative_to(storage.dir).as_posix(),
            }
            f.write(json.dumps(row) + "\n")


def test_floor_statistics_on_hand_made_runs(tmp_path: Path) -> None:
    from islands_harness.provenance import FloorIncomplete, analyze_floor

    cfg = _small(load_config(CONFIG), fixed=3, block=2, varied=3)
    model = cfg.model_by_id("qwen3-4b-local").model_copy(update={"documents": 4})
    plan = noise_floor_specs(cfg, model, DOCS, "A")
    plan = {"fixed": plan["fixed"], "varied": plan["varied"]}
    fixed_success = [[True, False], [True, False], [True, True]]
    fixed_texts = [["a", "b"], ["a", "b"], ["a", "c"]]  # rerun 3 differs on document 2
    varied_success = [
        [True, True, False, False],
        [True, True, True, False],
        [True, False, False, False],
    ]
    for r in range(3):
        _store(tmp_path, plan["fixed"][r], fixed_success[r], fixed_texts[r])
        _store(tmp_path, plan["varied"][r], varied_success[r], ["x"] * 4)

    fixed, varied = analyze_floor(tmp_path, plan, regime="sweep_regime_round_robin")
    assert fixed.protocol == "fixed_seed" and (fixed.reruns, fixed.n_per_rerun) == (3, 2)
    assert fixed.identity_rate == 0.5  # rerun 2 matches rerun 1, rerun 3 does not
    assert (fixed.flipped_documents, fixed.flip_rate) == (1, 0.5)
    assert fixed.rates == [0.5, 0.5, 1.0]
    assert fixed.jitter_sd == pytest.approx(statistics.stdev([0.5, 0.5, 1.0]))
    assert fixed.binomial_sd is None and fixed.phi is None
    assert 0.0 < fixed.flip_lo < 0.5 < fixed.flip_hi < 1.0
    assert varied.protocol == "varied_seed" and varied.identity_rate is None
    assert varied.rates == [0.5, 0.75, 0.25] and varied.mean_rate == 0.5
    assert varied.jitter_sd == pytest.approx(0.25)
    assert varied.binomial_sd == pytest.approx(math.sqrt(0.5 * 0.5 / 4))
    assert varied.phi == pytest.approx(1.0)
    assert (varied.flipped_documents, varied.aborted) == (2, 0)

    # A run aborted twice leaves every rate and counts as aborted.
    rows = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    last = json.loads(rows[-1])
    assert last["run_id"] == plan["varied"][2][3].run_id
    rows[-1] = json.dumps({**last, "success": None})
    (tmp_path / "runs.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    _, varied = analyze_floor(tmp_path, plan, regime="r")
    assert varied.aborted == 1 and varied.rates[2] == pytest.approx(1 / 3)

    # An incomplete floor is refused, not analysed.
    (tmp_path / "runs.jsonl").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(FloorIncomplete, match="1 of 18 planned"):
        analyze_floor(tmp_path, plan, regime="r")


def test_noise_floor_end_to_end_on_the_synthetic_agent(tmp_path: Path) -> None:
    from islands_harness.provenance import noise_floor, rebuild_spec, selftest_model
    from islands_harness.providers.synthetic import SyntheticAgent
    from islands_harness.runner import load_task

    cfg = _small(load_config(CONFIG), fixed=3, block=4, varied=4)
    model = selftest_model(cfg).model_copy(update={"documents": 8})
    task = load_task(cfg, REPO)
    agent = SyntheticAgent(
        {d: g.fields for d, g in task.gold.items()}, task.documents, model_id=model.id
    )
    out = tmp_path / model.id
    results = noise_floor(cfg, model, agent, repo_root=REPO, out_dir=out)
    fixed, varied = results
    # The synthetic agent draws from the sampling seed only: fixed seeds reproduce exactly.
    assert (fixed.protocol, fixed.reruns, fixed.n_per_rerun) == ("fixed_seed", 3, 4)
    assert (fixed.identity_rate, fixed.flip_rate, fixed.jitter_sd) == (1.0, 0.0, 0.0)
    assert (varied.protocol, varied.reruns, varied.n_per_rerun) == ("varied_seed", 4, 8)
    assert len(varied.rates) == 4 and varied.identity_rate is None
    rows = [
        json.loads(x)
        for x in (out / "floor" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3 * 4 + 4 * 8
    assert {r["phase"] for r in rows} == {"floor_fixed", "floor_varied"}
    assert not (out / "runs.jsonl").exists()  # the sweep's file is untouched

    jitter = json.loads((out / "jitter.json").read_text(encoding="utf-8"))
    assert jitter["schema"] == "jitter.v1" and jitter["model_id"] == model.id
    assert jitter["regime"] == "sweep_regime_round_robin"
    assert (jitter["identity_rate"], jitter["flip_rate"], jitter["jitter_sd"]) == (1.0, 0.0, 0.0)
    assert jitter["phi"] == varied.phi and jitter["runs"] == 44
    assert [p["protocol"] for p in jitter["protocols"]] == ["fixed_seed", "varied_seed"]
    assert len(jitter["sessions"]) == 1
    assert set(jitter["sessions"][0]["harness"]) == {"commit", "dirty"}
    assert jitter["sessions"][0]["protocols"]["varied"]["executed"] == 32

    # Resume: a second call executes nothing and reproduces every number.
    assert noise_floor(cfg, model, agent, repo_root=REPO, out_dir=out) == results
    sessions = json.loads((out / "jitter.json").read_text(encoding="utf-8"))["sessions"]
    assert len(sessions) == 2 and sessions[1]["protocols"]["fixed"]["executed"] == 0
    assert noise_floor(cfg, model, repo_root=REPO, out_dir=out, execute=False) == results

    # Replay rebuilds a fixed-seed floor run's spec, rerun index included.
    row = next(r for r in rows if r["phase"] == "floor_fixed" and r["index"] == 2)
    spec = rebuild_spec(row, cfg, model, sorted(task.documents))
    assert spec.run_id == row["run_id"] and spec.rerun in range(3)
    row = next(r for r in rows if r["phase"] == "floor_varied" and r["seed_offset"] == 3)
    assert rebuild_spec(row, cfg, model, sorted(task.documents)).rerun == 3


def test_report_reads_the_floor_into_the_site_jitter_block(tmp_path: Path) -> None:
    from islands_harness.report import _jitter

    block = _jitter(
        {
            "identity_rate": 1.0,
            "flip_rate": 0.0,
            "flip_lo": 0.0,
            "flip_hi": 0.16,
            "jitter_sd": 0.0,
            "phi": 0.93,
            "regime": "sweep_regime_round_robin",
        }
    )
    assert block.identity_rate == 1.0 and block.phi == 0.93
    assert block.regime == "sweep_regime_round_robin"
