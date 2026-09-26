"""The reviewer's checks: re-analysis, re-grading, file hashes and re-execution
(ARCHITECTURE.md Sections 6 and 14).

Three tiers, in rising cost:

- **Re-analysis** (``check_analysis``, also ``analyze --check``): recompute every analysis
  output from runs.jsonl and the pre-registration into a temporary directory and compare it
  with the published files. Identical bytes are expected on the same software stack; across
  machines, floating-point sums can differ in the last bits, so JSON and CSV numbers are also
  compared with a relative tolerance of 1e-9 and reported as "within tolerance".
- **Re-grading** (``regrade``): for every graded run, recover the accepted submission from
  the stored transcript, check it is the one runs.jsonl recorded, and score it again with
  the frozen grader.
- **Re-execution** (``reexecute``): run chosen cells again against a live model and compare
  the success counts, and the transcripts run by run, with the published ones.

``verify`` in the CLI runs the bundle hashes (SHA256SUMS), the freeze lock, re-analysis, and
optionally re-grading and re-execution, and exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import io
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from islands_harness.config import (
    ModelConfig,
    SnapshotConfig,
    load_prereg,
    prereg_hash,
)
from islands_harness.providers.base import ChatProvider

ANALYSIS_FILES = (
    "cells.csv",
    "results.json",
    "verdict.json",
    "statement.txt",
    "bootstrap/resamples.json",
)
REL_TOL = 1e-9
ABS_TOL = 1e-12
_MAX_DIFFERENCES = 20


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def compare_json(a: Any, b: Any, path: str = "$", out: list[str] | None = None) -> list[str]:
    """Paths where two JSON values differ, floats compared with the module tolerance, at
    most 20 entries."""
    out = [] if out is None else out
    if len(out) >= _MAX_DIFFERENCES:
        return out
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not _close(float(a), float(b)):
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}.{key}: present on one side only")
            else:
                compare_json(a[key], b[key], f"{path}.{key}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items != {len(b)} items")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            compare_json(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out[:_MAX_DIFFERENCES]


def compare_csv(a: str, b: str) -> list[str]:
    """Cells where two CSV texts differ, numbers compared with the module tolerance."""
    rows_a = list(csv.reader(io.StringIO(a)))
    rows_b = list(csv.reader(io.StringIO(b)))
    out: list[str] = []
    if len(rows_a) != len(rows_b):
        out.append(f"{len(rows_a)} rows != {len(rows_b)} rows")
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b, strict=False)):
        if len(ra) != len(rb):
            out.append(f"row {i}: {len(ra)} columns != {len(rb)} columns")
            continue
        for j, (x, y) in enumerate(zip(ra, rb, strict=True)):
            if x == y:
                continue
            try:
                same = _close(float(x), float(y))
            except ValueError:
                same = False
            if not same:
                out.append(f"row {i} column {j}: {x!r} != {y!r}")
        if len(out) >= _MAX_DIFFERENCES:
            break
    return out[:_MAX_DIFFERENCES]


@dataclass(frozen=True)
class FileCheck:
    name: str
    status: str  # identical | within_tolerance | different | missing
    differences: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("identical", "within_tolerance")


def compare_file(name: str, published: Path, recomputed: Path) -> FileCheck:
    if not published.is_file():
        return FileCheck(name, "missing", [f"{published} does not exist"])
    a, b = published.read_bytes(), recomputed.read_bytes()
    if a == b:
        return FileCheck(name, "identical")
    text_a, text_b = a.decode("utf-8"), b.decode("utf-8")
    if name.endswith(".json"):
        diffs = compare_json(json.loads(text_a), json.loads(text_b))
    elif name.endswith(".csv"):
        diffs = compare_csv(text_a, text_b)
    else:
        diffs = (
            []
            if text_a.strip() == text_b.strip()
            else [f"{text_a.strip()!r} != {text_b.strip()!r}"]
        )
    return FileCheck(name, "within_tolerance" if not diffs else "different", diffs)


# --------------------------------------------------------------------------------------
# Re-analysis
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisCheck:
    files: list[FileCheck]
    prereg_matches: bool  # the current r4 hashes to the hash results.json records
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.prereg_matches and all(f.ok for f in self.files)


def check_analysis(results_dir: Path, *, config: SnapshotConfig, repo_root: Path) -> AnalysisCheck:
    """Recompute every analysis output of one model directory into a temporary directory,
    with the snapshot id, pairing label, mix and pre-registration hash that results.json
    records, and compare file by file."""
    from islands_harness.stats.analyze import analyze, read_runs

    results_dir = Path(results_dir)
    published = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    runs, torn = read_runs(results_dir / "runs.jsonl")
    prereg = load_prereg(repo_root / config.snapshot.preregistration)
    recorded_hash = str(published["preregistration"]["hash"])
    matches = prereg_hash(prereg) == recorded_hash
    with tempfile.TemporaryDirectory() as tmp:
        analyze(
            runs,
            prereg,
            snapshot_id=str(published["snapshot"]),
            model_id=str(published["model"]),
            pairing=str(published["recovery"]["pairing"]),
            prereg_hash=recorded_hash,
            out_dir=Path(tmp),
            mix=str(published["mix"]),
            torn_lines=torn,
        )
        files = [compare_file(n, results_dir / n, Path(tmp) / n) for n in ANALYSIS_FILES]
    detail = "" if matches else "the pre-registration on disk is not the one the analysis recorded"
    return AnalysisCheck(files, matches, detail)


# --------------------------------------------------------------------------------------
# Re-grading
# --------------------------------------------------------------------------------------


def accepted_submission(messages: list[dict[str, Any]]) -> tuple[bool, Any]:
    """The loop's acceptance rule read back from a stored transcript: the first submit_record
    result that was executed without a model error, and the record its call carried."""
    calls: dict[str, dict[str, Any]] = {}
    for m in messages:
        if m.get("role") == "assistant":
            calls = {str(c.get("id")): c for c in m.get("tool_calls") or []}
        for r in m.get("tool_results") or []:
            if r.get("name") == "submit_record" and r.get("executed") and not r.get("model_error"):
                call = calls.get(str(r.get("call_id")))
                if call is None:
                    return True, None
                return True, json.loads(call["arguments_text"]).get("record")
    return False, None


@dataclass(frozen=True)
class RegradeResult:
    checked: int
    score_mismatches: list[str]
    submission_mismatches: list[str]
    missing_transcripts: list[str]

    @property
    def ok(self) -> bool:
        return not (self.score_mismatches or self.submission_mismatches or self.missing_transcripts)


def regrade(results_dir: Path, *, config: SnapshotConfig, repo_root: Path) -> RegradeResult:
    """Score every graded run again from its transcript with the frozen grader and tau."""
    from islands_harness.runner import load_task, score_submission
    from islands_harness.stats.analyze import read_runs

    results_dir = Path(results_dir)
    runs, _ = read_runs(results_dir / "runs.jsonl")
    task = load_task(config, repo_root)
    tau = load_prereg(repo_root / config.snapshot.preregistration).success.tau
    score_bad, submission_bad, missing = [], [], []
    checked = 0
    for row in runs:
        if row.get("success") is None:
            continue
        path = results_dir / str(row.get("transcript_path", ""))
        if not path.is_file():
            missing.append(str(row["run_id"]))
            continue
        messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
        _, record = accepted_submission(messages)
        if record != row.get("submission"):
            submission_bad.append(str(row["run_id"]))
            continue
        score, success, _ = score_submission(
            task.grader, row.get("submission"), task.gold[row["doc_id"]].fields, tau
        )
        if not _close(score, float(row["score"])) or success != bool(row["success"]):
            score_bad.append(f"{row['run_id']}: recorded {row['score']}, regraded {score}")
        checked += 1
    return RegradeResult(checked, score_bad, submission_bad, missing)


# --------------------------------------------------------------------------------------
# Re-execution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CellReexecution:
    tools: int
    fault: float
    published_k: int
    published_n: int
    k: int
    n: int
    difference_interval: tuple[float, float]  # Newcombe, new minus published
    identical_transcripts: int
    compared_transcripts: int

    @property
    def consistent(self) -> bool:
        """The published and re-executed rates are not distinguishable by their intervals."""
        lo, hi = self.difference_interval
        return lo <= 0.0 <= hi


async def reexecute(
    results_dir: Path,
    cells: list[tuple[int, float]],
    provider: ChatProvider,
    *,
    config: SnapshotConfig,
    model: ModelConfig,
    repo_root: Path,
    out_dir: Path,
) -> list[CellReexecution]:
    """Run each cell again from its specs into ``out_dir`` (one subdirectory per cell) and
    compare with the published runs: success counts through a Newcombe interval, and the
    canonical transcript hash run by run. On a stack the gate verified, every transcript
    should be identical; elsewhere the counts are read against the published jitter."""
    from islands_harness.provenance import canonical_transcript_hash
    from islands_harness.runner import Storage, load_task, read_transcript, run_specs
    from islands_harness.specs import Cell
    from islands_harness.specs import run_specs as specs_for_cell
    from islands_harness.stats.analyze import read_runs
    from islands_harness.stats.fit import newcombe

    results_dir = Path(results_dir)
    published = {r["run_id"]: r for r in read_runs(results_dir / "runs.jsonl")[0]}
    mix = next(iter(published.values()))["mix"] if published else "A"
    task = load_task(config, repo_root)
    tau = load_prereg(repo_root / config.snapshot.preregistration).success.tau
    report: list[CellReexecution] = []
    for tools, fault in cells:
        specs = specs_for_cell(config, model, Cell(tools, fault, mix), sorted(task.documents))
        storage = Storage(Path(out_dir) / f"cell-{tools}-{int(round(fault * 100))}")
        await run_specs(
            specs, provider, task, storage, None, config=config, model=model, tau=tau, resume=False
        )
        new = {r["run_id"]: r for r in read_runs(storage.runs_path)[0]}
        old = [published[i] for i in new if i in published]
        k_old = sum(1 for r in old if r.get("success"))
        n_old = sum(1 for r in old if r.get("success") is not None)
        k_new = sum(1 for r in new.values() if r.get("success"))
        n_new = sum(1 for r in new.values() if r.get("success") is not None)
        same = compared = 0
        for run_id, row in new.items():
            if run_id not in published:
                continue
            a = results_dir / published[run_id]["transcript_path"]
            b = storage.dir / row["transcript_path"]
            if a.is_file() and b.is_file():
                compared += 1
                same += canonical_transcript_hash(read_transcript(a)) == canonical_transcript_hash(
                    read_transcript(b)
                )
        interval = newcombe(k_new, n_new, k_old, n_old) if n_new and n_old else (float("nan"),) * 2
        report.append(
            CellReexecution(tools, fault, k_old, n_old, k_new, n_new, interval, same, compared)
        )
    return report
