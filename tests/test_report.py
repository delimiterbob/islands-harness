"""Outputs and verification (M4): snapshot.json, heatmap, manifest, bundle, the reviewer's
checks, replay and re-execution, all on a small synthetic snapshot built once per module
from the synthetic agent (providers/synthetic.py) through the real runner and analysis."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from islands_harness.config import load_config, load_prereg, prereg_hash, sha256_bytes
from islands_harness.provenance import replay_async, selftest, selftest_model
from islands_harness.providers.synthetic import SyntheticAgent
from islands_harness.report import (
    SCHEMAS,
    RunRowV1,
    SnapshotV1,
    bundle,
    bundle_members,
    check_sums,
    pct_text,
    report,
    schema_text,
    write_heatmap_png,
)
from islands_harness.runner import Storage, load_task, read_transcript, run_specs
from islands_harness.specs import expand_cells
from islands_harness.specs import run_specs as specs_for_cell
from islands_harness.stats.analyze import analyze, read_runs
from islands_harness.verification import (
    accepted_submission,
    check_analysis,
    compare_json,
    reexecute,
    regrade,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "snapshot-1.yaml"
ENV = {
    "harness_version": "test",
    "commit": "test",
    "python": "3.12",
    "os": "test",
    "packages": {},
    "gpu": None,
    "llama_cpp": None,
}


@pytest.fixture(scope="module")
def setup() -> dict:
    cfg = load_config(CONFIG)
    model = selftest_model(cfg).model_copy(update={"documents": 10})
    task = load_task(cfg, REPO)
    agent = SyntheticAgent(
        {d: g.fields for d, g in task.gold.items()}, task.documents, model_id=model.id
    )
    return {"cfg": cfg, "model": model, "task": task, "agent": agent}


@pytest.fixture(scope="module")
def built(setup: dict, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 10-document synthetic snapshot: sweep, analysis and report, never modified."""
    cfg, model, task, agent = setup["cfg"], setup["model"], setup["task"], setup["agent"]
    prereg = load_prereg(REPO / cfg.snapshot.preregistration)
    specs = [
        s
        for cell in expand_cells(cfg, "A")
        for s in specs_for_cell(cfg, model, cell, sorted(task.documents))
    ]
    snap = tmp_path_factory.mktemp("snapshot")
    model_dir = snap / model.id
    asyncio.run(
        run_specs(
            specs,
            agent,
            task,
            Storage(model_dir),
            None,
            config=cfg,
            model=model,
            tau=prereg.success.tau,
            concurrency=1,
            resume=False,
        )
    )
    rows, torn = read_runs(model_dir / "runs.jsonl")
    analyze(
        rows,
        prereg,
        snapshot_id="test",
        model_id=model.id,
        pairing="exact_synthetic_agent",
        prereg_hash=prereg_hash(prereg),
        out_dir=model_dir,
        torn_lines=torn,
    )
    report(
        snap, config=cfg, repo_root=REPO, kind="selftest", models={model.id: model}, environment=ENV
    )
    return snap


@pytest.fixture
def snapshot(built: Path, tmp_path: Path) -> Path:
    """A private copy of the built snapshot, free to tamper with."""
    copy = tmp_path / "snapshot"
    shutil.copytree(built, copy)
    return copy


def _model_dir(snap: Path) -> Path:
    return next(d for d in snap.iterdir() if (d / "results.json").is_file())


# -- snapshot.json ------------------------------------------------------------------------


def test_snapshot_json_validates_and_restates_the_analysis(built: Path) -> None:
    data = json.loads((built / "snapshot.json").read_text(encoding="utf-8"))
    snap = SnapshotV1.model_validate(data)
    results = json.loads((_model_dir(built) / "results.json").read_text(encoding="utf-8"))
    entry = snap.models[0]
    assert snap.snapshot.kind == "selftest" and "not a measurement" in snap.snapshot.label
    assert entry.statement == results["statement"]
    assert entry.role == "synthetic" and entry.stack.label == "synthetic"
    assert entry.identity.gpu is None and entry.identity.fingerprints == ["synthetic-v1"]
    assert len(entry.grid.cells) == 36 and sum(not c.in_scope for c in entry.grid.cells) == 6
    assert entry.verdicts.p1.verdict == results["verdicts"]["p1"]["verdict"]
    assert entry.fit.estimator == results["fit"]["status"]
    assert entry.summary.viable_fraction.descriptive is True
    assert entry.files.runs_jsonl.endswith("/runs.jsonl")
    for name in ("heatmap.svg", "manifest.json"):
        assert (_model_dir(built) / name).is_file()
    assert (built / "statement.txt").read_text(encoding="utf-8").strip() == entry.statement


def test_manifest_records_identities_counts_and_file_hashes(built: Path) -> None:
    d = _model_dir(built)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    rows, _ = read_runs(d / "runs.jsonl")
    assert manifest["counts"]["runs"] == len(rows)
    assert sum(manifest["counts"]["by_outcome"].values()) == len(rows)
    assert manifest["identities"]["fingerprints"]["synthetic-v1"]["turns"] > len(rows)
    assert manifest["files"]["runs.jsonl"] == sha256_bytes((d / "runs.jsonl").read_bytes())
    assert manifest["lock"] is None  # a selftest is not frozen


def test_heatmap_and_page_round_half_up() -> None:
    assert [pct_text(p) for p in (0.925, 0.915, 0.5, 0.004, 0.995)] == [
        "93",
        "92",
        "50",
        "0",
        "100",
    ]


def test_every_run_row_validates_against_runs_v1(built: Path) -> None:
    rows, _ = read_runs(_model_dir(built) / "runs.jsonl")
    for row in rows:
        RunRowV1.model_validate(row)


def test_exported_schemas_match_the_models() -> None:
    """schemas/ is committed; it must be regenerated (`islands schema`) when a model changes."""
    for name, model in SCHEMAS.items():
        path = REPO / "schemas" / f"{name}.json"
        assert path.read_text(encoding="utf-8") == schema_text(model), f"{name} is stale"


# -- bundle -------------------------------------------------------------------------------


def test_bundle_is_reproducible_and_its_sums_catch_tampering(snapshot: Path) -> None:
    first = bundle(snapshot)
    sums_a = (snapshot / "SHA256SUMS").read_bytes()
    with zipfile.ZipFile(first.path) as zf:
        names = zf.namelist()
        assert names == [*bundle_members(snapshot), "SHA256SUMS"]
        assert {i.date_time for i in zf.infolist()} == {(1980, 1, 1, 0, 0, 0)}
    second = bundle(snapshot)
    assert (snapshot / "SHA256SUMS").read_bytes() == sums_a and second.sha256 == first.sha256
    assert check_sums(snapshot) == {"mismatched": [], "missing": [], "unlisted": []}
    with (_model_dir(snapshot) / "cells.csv").open("a", encoding="utf-8") as f:
        f.write("tampered\n")
    (snapshot / "extra.txt").write_text("added later", encoding="utf-8")
    sums = check_sums(snapshot)
    assert sums["mismatched"] == [f"{_model_dir(snapshot).name}/cells.csv"]
    assert sums["unlisted"] == ["extra.txt"]


# -- re-analysis and re-grading ----------------------------------------------------------


def test_check_analysis_identical_then_tolerant_then_different(setup: dict, snapshot: Path) -> None:
    cfg, d = setup["cfg"], _model_dir(snapshot)
    assert all(f.status == "identical" for f in check_analysis(d, config=cfg, repo_root=REPO).files)
    results = json.loads((d / "results.json").read_text(encoding="utf-8"))
    results["fit"]["coefficients"]["tc"] *= 1 + 1e-12  # the last bits, as another BLAS might
    (d / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    check = check_analysis(d, config=cfg, repo_root=REPO)
    assert (
        check.ok and {f.name: f.status for f in check.files}["results.json"] == "within_tolerance"
    )
    results["fit"]["coefficients"]["tc"] += 0.1
    (d / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    check = check_analysis(d, config=cfg, repo_root=REPO)
    bad = {f.name: f for f in check.files}["results.json"]
    assert not check.ok and bad.status == "different"
    assert any("coefficients.tc" in line for line in bad.differences)


def test_compare_json_reports_paths() -> None:
    assert compare_json({"a": [1, 2.0]}, {"a": [1, 2.0 + 1e-15]}) == []
    assert compare_json({"a": True}, {"a": 1}) == ["$.a: True != 1"]
    assert compare_json({"a": 1}, {"b": 1}) == [
        "$.a: present on one side only",
        "$.b: present on one side only",
    ]


def test_regrade_finds_a_changed_score_and_a_changed_submission(
    setup: dict, snapshot: Path
) -> None:
    cfg, d = setup["cfg"], _model_dir(snapshot)
    clean = regrade(d, config=cfg, repo_root=REPO)
    assert clean.ok and clean.checked > 0
    lines = (d / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    failed = next(i for i, r in enumerate(rows) if r["success"] is False)
    submitted = next(i for i, r in enumerate(rows) if r["outcome"] == "submitted" and i != failed)
    rows[failed]["score"], rows[failed]["success"] = 1.0, True
    rows[submitted]["submission"] = {**rows[submitted]["submission"], "currency": "XXX"}
    (d / "runs.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    result = regrade(d, config=cfg, repo_root=REPO)
    assert not result.ok
    assert [m.split(":")[0] for m in result.score_mismatches] == [rows[failed]["run_id"]]
    assert result.submission_mismatches == [rows[submitted]["run_id"]]


def test_accepted_submission_follows_the_loop_rule() -> None:
    call = {"id": "c1", "name": "submit_record", "arguments_text": '{"record": {"a": 1}}'}
    messages = [
        {"role": "assistant", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_results": [
                {"call_id": "c1", "name": "submit_record", "executed": False, "model_error": False}
            ],
        },
        {"role": "assistant", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_results": [
                {"call_id": "c1", "name": "submit_record", "executed": True, "model_error": False}
            ],
        },
    ]
    assert accepted_submission(messages) == (True, {"a": 1})
    assert accepted_submission(messages[:2]) == (False, None)


# -- replay and re-execution -------------------------------------------------------------


def test_stored_transcripts_hash_like_the_live_ones(setup: dict, built: Path) -> None:
    from islands_harness.provenance import canonical_transcript_hash, rebuild_spec
    from islands_harness.runner import execute_spec

    cfg, model, task, agent = setup["cfg"], setup["model"], setup["task"], setup["agent"]
    d = _model_dir(built)
    row = next(r for r in read_runs(d / "runs.jsonl")[0] if r.get("fault_events"))
    spec = rebuild_spec(row, cfg, model, sorted(task.documents))
    _, live = asyncio.run(execute_spec(spec, agent, task, config=cfg, model=model))
    stored = read_transcript(d / row["transcript_path"])
    assert canonical_transcript_hash(live) == canonical_transcript_hash(stored)


def test_replay_is_identical_and_reports_where_a_different_agent_diverges(
    setup: dict, built: Path
) -> None:
    cfg, model, task = setup["cfg"], setup["model"], setup["task"]
    d = _model_dir(built)
    rows = read_runs(d / "runs.jsonl")[0]
    row = next(
        r
        for r in rows
        if r["outcome"] == "submitted" and int(r["tools"]) == 2 and not r["fault_events"]
    )
    same = asyncio.run(
        replay_async(row["run_id"], d, setup["agent"], config=cfg, repo_root=REPO, model=model)
    )
    assert same.identical and same.first_divergence is None and same.replay_outcome == "submitted"

    class Fumbler(SyntheticAgent):  # always gives up after the fetch
        async def complete(self, messages, tools, sampling):  # noqa: ANN001, ANN201
            if any(m.role == "tool" for m in messages):
                return self._text(messages, 1, "x", "I give up.")
            return await super().complete(messages, tools, sampling)

    other = Fumbler({k: g.fields for k, g in task.gold.items()}, task.documents, model_id=model.id)
    diff = asyncio.run(
        replay_async(row["run_id"], d, other, config=cfg, repo_root=REPO, model=model)
    )
    assert not diff.identical and diff.first_divergence == 4  # system, user, fetch, result, then
    assert diff.replay_outcome == "no_submission"


def test_rebuild_spec_refuses_a_changed_config(setup: dict, built: Path) -> None:
    from islands_harness.provenance import rebuild_spec

    cfg, model, task = setup["cfg"], setup["model"], setup["task"]
    row = read_runs(_model_dir(built) / "runs.jsonl")[0][0]
    moved = cfg.model_copy(update={"snapshot": cfg.snapshot.model_copy(update={"root_seed": 1})})
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        rebuild_spec(row, moved, model, sorted(task.documents))


def test_reexecution_matches_the_published_cell(setup: dict, built: Path, tmp_path: Path) -> None:
    cfg, model = setup["cfg"], setup["model"]
    report_rows = asyncio.run(
        reexecute(
            _model_dir(built),
            [(3, 0.2), (1, 0.0)],
            setup["agent"],
            config=cfg,
            model=model,
            repo_root=REPO,
            out_dir=tmp_path / "again",
        )
    )
    for r in report_rows:
        assert r.consistent and (r.k, r.n) == (r.published_k, r.published_n)
        assert r.identical_transcripts == r.compared_transcripts == r.n


# -- the commands ------------------------------------------------------------------------


def test_verify_command_passes_then_fails_on_a_tampered_copy(snapshot: Path) -> None:
    from islands_harness.cli import main

    bundle(snapshot)
    args = ["verify", str(snapshot), "--exploratory", "--regrade", "-c", str(CONFIG)]
    assert main(args) == 0
    runs = _model_dir(snapshot) / "runs.jsonl"
    rows = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
    i = next(i for i, r in enumerate(rows) if r["success"] is False)
    rows[i]["success"], rows[i]["score"] = True, 1.0
    runs.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    assert main(args) == 1


def test_analyze_check_command(snapshot: Path) -> None:
    from islands_harness.cli import main

    d = _model_dir(snapshot)
    assert main(["analyze", str(d), "--check", "-c", str(CONFIG)]) == 0
    (d / "statement.txt").write_text("edited\n", encoding="utf-8")
    assert main(["analyze", str(d), "--check", "-c", str(CONFIG)]) == 1


def test_png_needs_the_plots_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    grid = {"tools_axis": [2], "fault_axis": [0.0], "cells": []}
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(RuntimeError, match="plots extra"):
        write_heatmap_png(grid, tmp_path / "h.png")


def test_png_renders_when_matplotlib_is_installed(built: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    entry = SnapshotV1.model_validate(
        json.loads((built / "snapshot.json").read_text(encoding="utf-8"))
    ).models[0]
    out = write_heatmap_png(entry.grid.model_dump(), tmp_path / "h.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_selftest_builds_and_checks_a_whole_snapshot(tmp_path: Path) -> None:
    summary = selftest(load_config(CONFIG), tmp_path / "selftest", REPO)
    snap = summary["snapshot"]
    assert snap["runs"] == 1320 and snap["regraded"] == 1320 and snap["replays"] == 4
    assert "P1: survives" in snap["statement"] and "P2: survives" in snap["statement"]
    data = json.loads(Path(snap["snapshot_json"]).read_text(encoding="utf-8"))
    assert SnapshotV1.model_validate(data).snapshot.kind == "selftest"


def test_export_bundle_shortens_paths_and_keeps_the_analysis_inputs(tmp_path: Path) -> None:
    """A deposit export of a results directory: the gate record and the session log lose the
    machine's paths, runs.jsonl is copied byte for byte, the sums check, and a path that
    still names the user (a Windows short name) stops the export before it writes."""
    import json

    from islands_harness.report import REDACTIONS_NAME, check_sums, export_bundle

    base = tmp_path.resolve()
    home = base / "robertsmith"
    root = home / "work" / "harness"
    results = root / "results" / "s1" / "m"
    (results / "floor").mkdir(parents=True)
    exe = home / "AppData" / "Local" / "islands" / "llama-server.exe"
    gate = {"server": {"command_line": [str(exe), "-m", str(root / "models" / "m.gguf")]}}
    (results / "determinism.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    session = {"server": {"model_path": str(root / "models" / "m.gguf")}}
    (results / "floor" / "sessions.jsonl").write_text(json.dumps(session) + "\n", encoding="utf-8")
    runs = '{"run_id": "a", "doc_id": "inv-0001"}\n'
    (results / "runs.jsonl").write_text(runs, encoding="utf-8")

    out = base / "export"
    result = export_bundle(results, out, repo_root=root, home=home)
    assert json.loads((out / "determinism.json").read_text(encoding="utf-8"))["server"][
        "command_line"
    ][0] == str(exe).replace(str(home), "~", 1)
    assert "robertsmith" not in (out / "floor" / "sessions.jsonl").read_text(encoding="utf-8")
    assert (out / "runs.jsonl").read_bytes() == (results / "runs.jsonl").read_bytes()
    listed = (out / REDACTIONS_NAME).read_text(encoding="utf-8")
    assert listed.endswith("  determinism.json\n  floor/sessions.jsonl\n")
    assert check_sums(out)["mismatched"] == [] and result.members == 5
    assert "robertsmith" in (results / "determinism.json").read_text(
        encoding="utf-8"
    )  # source kept

    (results / "notes.txt").write_text(str(base / "ROBERT~1" / "x") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names the local user"):
        export_bundle(results, base / "export2", repo_root=root, home=home)
    assert not (base / "export2").exists()
