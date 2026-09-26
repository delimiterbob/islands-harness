"""Calibration and the freeze (M6): pilot seeds, rate tables, the joint recommendation, the
pilot itself against the synthetic agent, the pre-registration hash that leaves deposit
metadata out, and a freeze that refuses while anything is still marked for confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from islands_harness import dataset as ds
from islands_harness.config import load_config, load_prereg, prereg_hash
from islands_harness.provenance import calibrate, pilot_seed, rate_table, recommend_calibration
from islands_harness.providers.synthetic import SyntheticAgent

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "snapshot-1.yaml"


@pytest.fixture(scope="module")
def cfg():  # noqa: ANN201
    return load_config(CONFIG)


def test_pilot_seeds_are_stable_and_never_the_snapshot_seed(cfg) -> None:  # noqa: ANN001
    seeds = [pilot_seed(cfg, d) for d in (1, 2, 3)]
    assert seeds == [pilot_seed(cfg, d) for d in (1, 2, 3)] and len(set(seeds)) == 3
    snapshot_seed = json.loads(
        (REPO / "tasks/invoice_extraction/dataset/LOCK.json").read_text(encoding="utf-8")
    )["seed"]
    assert snapshot_seed not in seeds


def test_rate_table_counts_scores_against_each_tau() -> None:
    rows = [{"score": s} for s in (1.0, 0.9, 0.8, 0.7, 0.6)] + [{"score": None}]
    table = rate_table(rows, [0.8, 0.9, 0.7], (0.7, 0.9))
    assert (table["0.8"]["k"], table["0.8"]["n"]) == (3, 5)
    assert table["0.9"]["k"] == 2 and table["0.7"]["k"] == 4
    assert table["0.8"]["in_band"] is False and table["0.7"]["in_band"] is True  # 0.6 and 0.8
    assert table["0.8"]["lo"] < 0.6 < table["0.8"]["hi"]


def _result(model: str, rates: dict[tuple[int, float], float], malformed: float = 0.0) -> dict:
    levels: dict[str, dict] = {}
    for (level, tau), p in rates.items():
        entry = levels.setdefault(str(level), {"eligible": malformed <= 0.2, "taus": {}})
        entry["taus"][f"{tau:g}"] = {
            "p": p,
            "lo": p - 0.1,
            "hi": p + 0.1,
            "in_band": 0.7 <= p <= 0.9,
        }
    return {"model": model, "levels": levels}


def test_recommendation_prefers_replicates_then_tau_then_the_band_midpoint(cfg) -> None:  # noqa: ANN001
    cfg = cfg.model_copy(  # the ranking logic, under the r3 band and three levels
        update={
            "calibration": cfg.calibration.model_copy(
                update={"target_band": (0.7, 0.9), "difficulty_levels": [1, 2, 3]}
            )
        }
    )
    taus = cfg.calibration.tau_candidates  # 0.8, 0.9, 0.7
    a = {(d, t): 1.0 for d in (1, 2, 3) for t in taus}
    b = {(d, t): 1.0 for d in (1, 2, 3) for t in taus}
    a[(3, 0.8)], b[(3, 0.8)] = 0.95, 0.85  # tau 0.8 at level 3: one model in band
    a[(3, 0.9)], b[(3, 0.9)] = 0.80, 0.75  # tau 0.9 at level 3: both in band
    rec = recommend_calibration([_result("a", a), _result("b", b)], cfg)
    assert (rec["chosen"]["difficulty"], rec["chosen"]["tau"]) == (3, 0.9)
    assert rec["chosen"]["replicates"] == ["a", "b"] and rec["chosen"]["excluded"] == []
    a[(2, 0.8)], b[(2, 0.8)] = 0.72, 0.88
    a[(1, 0.8)], b[(1, 0.8)] = 0.80, 0.82  # both levels 1 and 2 in band at tau 0.8
    rec = recommend_calibration([_result("a", a), _result("b", b)], cfg)
    assert (rec["chosen"]["difficulty"], rec["chosen"]["tau"]) == (1, 0.8)  # closer to 0.80


def test_recommendation_excludes_a_model_above_the_malformed_limit(cfg) -> None:  # noqa: ANN001
    rates = {(d, t): 0.8 for d in (1, 2, 3) for t in cfg.calibration.tau_candidates}
    rec = recommend_calibration(
        [_result("ok", rates), _result("parser", rates, malformed=0.3)], cfg
    )
    assert rec["chosen"]["replicates"] == ["ok"] and rec["chosen"]["excluded"] == ["parser"]


def test_calibrate_runs_the_pilot_and_writes_calibration_json(cfg, tmp_path: Path) -> None:  # noqa: ANN001
    model = cfg.model_by_id("gpt-oss-20b-local")
    data = tmp_path / "pilot-d1" / "dataset"
    ds.generate(pilot_seed(cfg, 1), cfg.calibration.pilot_runs, 1, data)
    gold = {g.doc_id: g.fields for g in ds.load_gold(data).values()}
    agent = SyntheticAgent(gold, {d.id: d.text for d in ds.load(data)}, model_id=model.id)
    result = calibrate(cfg, model, agent, repo_root=REPO, out_dir=tmp_path, levels=[1])
    level = result["levels"]["1"]
    assert level["runs"] == cfg.calibration.pilot_runs and level["pilot_seed"] == pilot_seed(cfg, 1)
    assert level["eligible"] and set(level["taus"]) == {"0.8", "0.9", "0.7"}
    stored = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert stored["levels"]["1"]["taus"]["0.8"]["k"] == level["taus"]["0.8"]["k"]
    again = calibrate(cfg, model, agent, repo_root=REPO, out_dir=tmp_path, levels=[1])  # resumes
    assert again["levels"]["1"]["runs"] == level["runs"]


def test_prereg_hash_ignores_deposit_metadata_but_not_rules() -> None:
    prereg = load_prereg(REPO / "configs" / "preregistration" / "r4.yaml")
    base = prereg_hash(prereg)
    deposited = prereg.model_copy(
        update={"frozen_on": "2026-10-06", "deposit_doi": "10.5281/zenodo.1"}
    )
    assert prereg_hash(deposited) == base
    stricter = prereg.model_copy(update={"success": prereg.success.model_copy(update={"tau": 0.9})})
    assert prereg_hash(stricter) != base


def test_freeze_refuses_while_items_await_confirmation(tmp_path: Path) -> None:
    """In a scratch copy of the repository (never the real one): a marker, even one that
    wraps across two lines of prose, stops the freeze before any git or lock step."""
    import shutil

    from islands_harness.cli import main, unconfirmed_items

    root = tmp_path / "repo"
    for rel in ("configs/preregistration/r4.yaml", "configs/snapshot-1.yaml", "PREREGISTRATION.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    prose = root / "PREREGISTRATION.md"
    prose.write_text(
        prose.read_text(encoding="utf-8") + "\nOne more rule. **CONFIRM BEFORE\n  FREEZE.**\n",
        encoding="utf-8",
    )
    config = root / "configs" / "snapshot-1.yaml"

    class Args:
        pass

    args = Args()
    args.config = str(config)
    pending = unconfirmed_items(args, root)  # type: ignore[arg-type]
    assert len(pending) == 1 and pending[0].startswith("PREREGISTRATION.md:")
    assert main(["freeze", "-c", str(config)]) == 2
    assert not (root / "configs" / "snapshot-1.lock.json").exists()


def test_deposit_gathers_the_frozen_rules_and_records(cfg, tmp_path: Path) -> None:  # noqa: ANN001
    """A deposit built in a scratch copy of the repository, with a stand-in lock."""
    import shutil

    from islands_harness.report import check_sums, prepare_deposit

    root = tmp_path / "repo"
    for rel in (
        "configs/preregistration/r4.yaml",
        "configs/snapshot-1.yaml",
        "PREREGISTRATION.md",
        "docs/method-note.md",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    config_path = root / "configs" / "snapshot-1.yaml"
    with pytest.raises(FileNotFoundError, match="freeze lock"):
        prepare_deposit(cfg, root, config_path, tmp_path / "deposit")
    (root / "configs" / "snapshot-1.lock.json").write_text('{"stand": "in"}\n', encoding="utf-8")
    gate = root / "results" / "s1" / "gpt-oss-20b-local" / "determinism.json"
    gate.parent.mkdir(parents=True)
    gate.write_text('{"label": "deterministic_verified"}\n', encoding="utf-8")
    result = prepare_deposit(cfg, root, config_path, tmp_path / "deposit")
    sums = (tmp_path / "deposit" / "SHA256SUMS").read_text(encoding="utf-8")
    for rel in (
        "configs/preregistration/r4.yaml",
        "configs/snapshot-1.lock.json",
        "results/s1/gpt-oss-20b-local/determinism.json",
    ):
        assert f"  {rel}\n" in sums
    assert result.members == 7 and check_sums(tmp_path / "deposit")["mismatched"] == []


def test_deposit_shortens_local_paths(cfg, tmp_path: Path) -> None:  # noqa: ANN001
    """The gate records name the server binary and the weights by absolute path. The deposit
    copies carry them from "." (the repository root) or "~" (the home directory), stay valid
    JSON, and are refused while a path still names the user, here as a Windows short name."""
    import json
    import shutil

    from islands_harness.report import REDACTIONS_NAME, check_sums, prepare_deposit

    base = tmp_path.resolve()
    home = base / "robertsmith"
    root = home / "work" / "harness"
    for rel in ("configs/preregistration/r4.yaml", "configs/snapshot-1.yaml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    config_path = root / "configs" / "snapshot-1.yaml"
    (root / "configs" / "snapshot-1.lock.json").write_text('{"stand": "in"}\n', encoding="utf-8")
    exe = home / "AppData" / "Local" / "islands" / "llama-server.exe"
    weights = root / "models" / "weights" / "m.gguf"
    gate = root / "results" / "s1" / "gpt-oss-20b-local" / "determinism.json"
    gate.parent.mkdir(parents=True)
    server = {"command_line": [str(exe), "-m", str(weights).upper()], "model_path": str(weights)}
    gate.write_text(json.dumps({"server": server}, indent=2) + "\n", encoding="utf-8")
    note = root / "docs" / "method-note.md"
    note.parent.mkdir()
    note.write_text(f"Weights at {weights.as_posix()}; root '{root}'.\n", encoding="utf-8")

    out = base / "deposit"
    prepare_deposit(cfg, root, config_path, out, home=home)
    copied = json.loads((out / "results/s1/gpt-oss-20b-local/determinism.json").read_text("utf-8"))
    assert copied["server"] == {
        "command_line": [
            str(exe).replace(str(home), "~", 1),
            "-m",
            str(weights).upper().replace(str(root).upper(), ".", 1),
        ],
        "model_path": str(weights).replace(str(root), ".", 1),
    }
    prose = (out / "docs" / "method-note.md").read_text(encoding="utf-8")
    assert prose == f"Weights at {weights.as_posix().replace(root.as_posix(), '.', 1)}; root '.'.\n"
    listed = (out / REDACTIONS_NAME).read_text(encoding="utf-8")
    assert listed.endswith(
        "  docs/method-note.md\n  results/s1/gpt-oss-20b-local/determinism.json\n"
    )
    assert check_sums(out)["mismatched"] == []
    for f in out.rglob("*"):
        if f.is_file() and f.suffix != ".zip":
            assert "robertsmith" not in f.read_text(encoding="utf-8").lower(), f

    temp = base / "ROBERT~1" / "AppData" / "Local" / "Temp" / "run.log"
    gate.write_text(json.dumps({"server": {"log": str(temp)}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names the local user"):
        prepare_deposit(cfg, root, config_path, base / "deposit2", home=home)
    assert not (base / "deposit2").exists()


def test_recommendation_uses_the_current_band_not_the_stored_flags(cfg) -> None:  # noqa: ANN001
    """A pilot judged under an older band keeps its rates; the recommendation re-judges them."""
    rates = {(d, t): 1.0 for d in (1, 2, 3, 4) for t in cfg.calibration.tau_candidates}
    rates[(4, 0.8)] = 0.40
    stale = _result("m", rates)
    for level in stale["levels"].values():  # stored as if judged under [0.70, 0.90]
        for cell in level["taus"].values():
            cell["in_band"] = 0.7 <= cell["p"] <= 0.9
    wide = cfg.model_copy(
        update={
            "calibration": cfg.calibration.model_copy(
                update={"target_band": (0.15, 0.95), "difficulty_levels": [1, 2, 3, 4]}
            )
        }
    )
    rec = recommend_calibration([stale], wide)
    assert (rec["chosen"]["difficulty"], rec["chosen"]["tau"]) == (4, 0.8)
    assert rec["chosen"]["replicates"] == ["m"]
