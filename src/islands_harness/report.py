"""Site JSON, heatmap, statement, manifest, schema export and the Zenodo bundle.

``snapshot.json`` (schema snapshot.v1, ARCHITECTURE.md Section 12) is the one file the
website reads; ``SnapshotV1`` below is its definition and ``export_schemas`` writes the JSON
Schema the site can validate against. ``report`` builds it from the analysis outputs of one
model directory, or of every model directory under a snapshot directory, and writes beside
it a heatmap, a manifest per model and the combined statement.

Conventions shared with results.json and verdict.json: an interval end that is unbounded is
the string "inf" or "-inf", a quantity that could not be computed is null, and every number
that decides nothing is wrapped as ``{"value": ..., "descriptive": true}``.

The heatmap is pure SVG in the geometry of the mock's ``map.html`` grid (viewBox 0 0 384 380,
52 px cells on a 54 px pitch) and colours cells with the site's CSS variables, each with a
fallback so the file also renders on its own.

``bundle`` zips a results directory for deposit with a SHA256SUMS file. Member order, times
and attributes are fixed, so the member hashes are identical on every platform; the zip's
own bytes can still differ between zlib builds, which is why SHA256SUMS, not the zip hash,
is the reference a reviewer checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from islands_harness.config import (
    ModelConfig,
    SnapshotConfig,
    canonical_bytes,
    load_prereg,
    sha256_bytes,
)

# --------------------------------------------------------------------------------------
# snapshot.v1
# --------------------------------------------------------------------------------------

Num = float | Literal["inf", "-inf"]  # an interval end; unbounded ends are strings


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreregRef(_Model):
    version: str
    doi: str | None
    hash: str


class HarnessRef(_Model):
    version: str
    commit: str  # "uncommitted" when the harness did not run from a git checkout
    config_hash: str


class SnapshotMetaV1(_Model):
    id: str
    kind: Literal["snapshot", "exploratory", "selftest"]
    label: str
    date_start: str | None
    date_end: str | None
    preregistration: PreregRef
    harness: HarnessRef
    bundle_doi: str | None


class IdentityV1(_Model):
    model: str
    revision: str | None
    weights_verified: bool | None
    server: str | None
    build: str | None  # llama.cpp build tag; null for hosted models
    gpu: str | None
    response_models: list[str]
    fingerprints: list[str]
    sampling: dict[str, Any]


class StackV1(_Model):
    deterministic_verified: bool | None  # set by the gate; null when the gate did not run
    label: str  # deterministic_verified | non_deterministic | gate_not_run | hosted | synthetic


class CellV1(_Model):
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


class GridV1(_Model):
    tools_axis: list[int]
    fault_axis: list[float]
    cells: list[CellV1]


class JitterV1(_Model):
    identity_rate: float | None
    flip_rate: float | None
    flip_lo: float | None
    flip_hi: float | None
    jitter_sd: float | None
    phi: float | None
    regime: str  # "not_measured" until the noise floor has run


class FitV1(_Model):
    estimator: Literal["mle", "firth", "not_estimable"]
    coefficients: dict[str, float]
    intervals: dict[str, list[Num]]  # the rung's decision intervals
    intervals_source: Literal["wald_cluster", "bootstrap", "none"]
    peak: float | None
    peak_interval: list[Num] | None
    drop: float | None  # percentage points
    drop_interval: list[Num] | None
    status: str  # "converged", or why the ladder moved or stopped


class Descriptive(_Model):
    value: Any
    descriptive: Literal[True] = True


class SummaryV1(_Model):
    viable_fraction: Descriptive
    upper_tool_edge: Descriptive
    boundary_width: Descriptive
    recovery_time: Descriptive
    fault_axis_shape: Descriptive


class ConditionV1(_Model):
    name: str
    value: Any
    passed: bool | None
    detail: str = ""


class PropositionVerdictV1(_Model):
    verdict: str
    moves: bool  # false when the instrument check failed: the category does not move
    conditions: list[ConditionV1]


class VerdictsV1(_Model):
    instrument: str
    propositions_move: bool
    p1: PropositionVerdictV1
    p2: PropositionVerdictV1


class ModelFilesV1(_Model):
    """Paths relative to snapshot.json."""

    runs_jsonl: str
    results: str
    verdict: str
    manifest: str
    heatmap: str


class ModelEntryV1(_Model):
    id: str
    role: str  # local | hosted | synthetic
    mix: str
    identity: IdentityV1
    stack: StackV1
    grid: GridV1
    jitter: JitterV1
    fit: FitV1
    summary: SummaryV1
    verdicts: VerdictsV1
    statement: str
    spend_usd: float | None
    files: ModelFilesV1


class LinksV1(_Model):
    method_note: str
    preregistration: str


class SnapshotV1(_Model):
    schema_version: Literal[1] = 1
    snapshot: SnapshotMetaV1
    models: list[ModelEntryV1]
    links: LinksV1


class RunRowV1(_Model):
    """One runs.jsonl line (runs.v1). Mirrors loop.RunRecord; runner.record_to_json produces it."""

    run_id: str
    model_id: str
    phase: str
    mix: str
    tools: int
    fault_rate: float
    index: int
    doc_id: str
    epoch: int
    sample_seed: int
    seed_offset: int | None
    tool_list: list[str]
    outcome: str
    score: float | None
    success: bool | None
    per_field: dict[str, float] | None
    turns: int
    tool_calls: int
    seconds: float
    usage: dict[str, int]
    cost_usd: float | None
    fault_events: list[dict[str, Any]]
    first_fault_turn: int | None
    recovered: bool | None
    submission: Any
    response_models: list[str]
    fingerprints: list[str]
    identity: dict[str, Any]
    transcript_path: str
    events: list[dict[str, Any]]
    transport_retries: int = 0
    malformed_calls: int = 0
    submission_attempted_in_text: bool | None = None


def export_schema() -> dict[str, Any]:
    return SnapshotV1.model_json_schema()


SCHEMAS: dict[str, type[BaseModel]] = {
    "config.v1": SnapshotConfig,
    "runs.v1": RunRowV1,
    "snapshot.v1": SnapshotV1,
}


def schema_text(model: type[BaseModel]) -> str:
    """The exported JSON Schema of one model, exactly as ``export_schemas`` writes it."""
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"


def export_schemas(out_dir: Path) -> dict[str, Path]:
    """Write config.v1, runs.v1 and snapshot.v1 JSON Schemas and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, model in SCHEMAS.items():
        path = out_dir / f"{name}.json"
        path.write_text(schema_text(model), encoding="utf-8", newline="\n")
        written[name] = path
    return written


# --------------------------------------------------------------------------------------
# Environment and identities
# --------------------------------------------------------------------------------------

_PACKAGES = (
    "islands-harness",
    "numpy",
    "scipy",
    "pandas",
    "statsmodels",
    "firthmodels",
    "pydantic",
    "httpx2",
    "anthropic",
)


def _version(package: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def git_commit(repo_root: Path) -> str:
    """HEAD of the checkout the harness runs from, or "uncommitted" outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"
    return out.stdout.strip() or "uncommitted"


def _gpu() -> dict[str, str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.stdout.strip().splitlines()[:1]
    if not first:
        return None
    name, _, driver = first[0].partition(",")
    return {"name": name.strip(), "driver": driver.strip()}


def _llama_build() -> dict[str, Any] | None:
    import os

    base = Path(os.environ.get("LOCALAPPDATA", "")) / "islands" / "llama.cpp"
    for build in sorted(base.glob("*/build.json")) if base.exists() else []:
        try:
            return json.loads(build.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def collect_environment(repo_root: Path) -> dict[str, Any]:
    """Where the report was produced: harness version and commit, Python, OS, package
    versions, GPU and driver, and the installed llama.cpp build. Machine-specific by nature,
    so ``verify`` never compares it byte for byte."""
    return {
        "harness_version": _version("islands-harness"),
        "commit": git_commit(repo_root),
        "python": platform.python_version(),
        "os": platform.platform(),
        "packages": {p: _version(p) for p in _PACKAGES},
        "gpu": _gpu(),
        "llama_cpp": _llama_build(),
    }


def request_dates(runs: list[dict[str, Any]]) -> list[str]:
    """Every model request date (YYYY-MM-DD) recorded in the runs, sorted."""
    return sorted(
        str(e["data"]["identity"]["request_date"])[:10]
        for r in runs
        for e in r.get("events") or []
        if e.get("kind") == "model_turn"
        and ((e.get("data") or {}).get("identity") or {}).get("request_date")
    )


def served_identities(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Every distinct served model string and fingerprint, with the number of turns that
    reported it and its first and last request dates (null when the runs carry no dates)."""
    out: dict[str, dict[str, dict[str, Any]]] = {"response_models": {}, "fingerprints": {}}
    for r in runs:
        for e in r.get("events") or []:
            if e.get("kind") != "model_turn":
                continue
            ident = (e.get("data") or {}).get("identity") or {}
            date = str(ident["request_date"])[:10] if ident.get("request_date") else None
            for key, field in (("response_models", "model"), ("fingerprints", "fingerprint")):
                value = ident.get(field)
                if not value:
                    continue
                slot = out[key].setdefault(
                    str(value), {"turns": 0, "first_seen": None, "last_seen": None}
                )
                slot["turns"] += 1
                if date is not None:
                    slot["first_seen"] = min(filter(None, [slot["first_seen"], date]))
                    slot["last_seen"] = max(filter(None, [slot["last_seen"], date]))
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def _strip_flag(block: Any) -> Any:
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if k != "descriptive"}
    return block


def _conditions(verdict: dict[str, Any]) -> list[ConditionV1]:
    return [
        ConditionV1(
            name=c["name"], value=c.get("value"), passed=c.get("passed"), detail=c.get("detail", "")
        )
        for c in verdict.get("conditions") or []
    ]


SYNTHETIC_MODEL = "synthetic"  # ModelConfig.model of the selftest's synthetic agent


def _role(model: ModelConfig | None, results: dict[str, Any]) -> str:
    """local, hosted or synthetic (the selftest's agent, which no server or GPU serves)."""
    from islands_harness.provenance import is_hosted

    if model is None:
        return "synthetic" if "synthetic" in str(results.get("model")) else "local"
    if model.model == SYNTHETIC_MODEL:
        return "synthetic"
    return "hosted" if is_hosted(model) else "local"


def _stack(role: str, determinism: dict[str, Any] | None) -> StackV1:
    if role == "hosted":
        return StackV1(deterministic_verified=None, label="hosted")
    if role == "synthetic":
        return StackV1(deterministic_verified=None, label="synthetic")
    if determinism is None:
        return StackV1(deterministic_verified=None, label="gate_not_run")
    label = str(determinism.get("label", "gate_not_run"))
    return StackV1(deterministic_verified=label == "deterministic_verified", label=label)


def _jitter(jitter: dict[str, Any] | None) -> JitterV1:
    if not jitter:
        return JitterV1(
            identity_rate=None,
            flip_rate=None,
            flip_lo=None,
            flip_hi=None,
            jitter_sd=None,
            phi=None,
            regime="not_measured",
        )
    keys = ("identity_rate", "flip_rate", "flip_lo", "flip_hi", "jitter_sd", "phi")
    return JitterV1(**{k: jitter.get(k) for k in keys}, regime=str(jitter.get("regime", "")))


def _identity(
    model: ModelConfig | None,
    results: dict[str, Any],
    runs: list[dict[str, Any]],
    repo_root: Path,
    environment: dict[str, Any],
    role: str,
) -> IdentityV1:
    served = {
        "response_models": sorted({m for r in runs for m in r.get("response_models") or []}),
        "fingerprints": sorted({f for r in runs for f in r.get("fingerprints") or []}),
    }
    if role == "synthetic":
        return IdentityV1(
            model=SYNTHETIC_MODEL,
            revision=None,
            weights_verified=None,
            server="synthetic agent (islands_harness.providers.synthetic)",
            build=None,
            gpu=None,
            sampling={},
            **served,
        )
    revision = None
    if model is not None and model.lock is not None and (repo_root / model.lock).exists():
        revision = json.loads((repo_root / model.lock).read_text(encoding="utf-8")).get("revision")
    expect = model.expect if model is not None else None
    server = expect.server if expect is not None else None
    if model is not None and model.provider == "anthropic":
        server = f"anthropic:{model.transport or 'anthropic'}"
    sampling: dict[str, Any] = {}
    if model is not None:
        sampling = {**model.sampling.model_dump(), "extra_body": model.extra_body}
    gpu = environment.get("gpu") if role == "local" else None
    return IdentityV1(
        model=model.model if model is not None else str(results.get("model")),
        revision=revision,
        weights_verified=None,
        server=server,
        build=expect.build if expect is not None else None,
        gpu=gpu.get("name") if isinstance(gpu, dict) else None,
        sampling=sampling,
        **served,
    )


def _fit(results: dict[str, Any]) -> FitV1:
    fit = results["fit"]
    p1 = results["verdicts"]["p1"]
    status = fit["status"]
    if status == "mle":
        intervals, source = fit.get("wald_intervals") or {}, "wald_cluster"
    elif status == "firth":
        intervals, source = (
            (results.get("bootstrap") or {}).get("coef_intervals") or {},
            "bootstrap",
        )
    else:
        intervals, source = {}, "none"
    return FitV1(
        estimator=status,
        coefficients=fit.get("coefficients") or {},
        intervals={k: list(v) for k, v in intervals.items()},
        intervals_source=source,
        peak=p1.get("peak"),
        peak_interval=p1.get("peak_interval"),
        drop=p1.get("drop_points"),
        drop_interval=p1.get("drop_interval"),
        status=fit.get("reason") or "converged",
    )


def _summary(results: dict[str, Any]) -> SummaryV1:
    desc = results.get("descriptive") or {}
    p1 = results["verdicts"]["p1"]
    rec = results.get("recovery") or {}
    shape = desc.get("fault_axis_shape") or {}
    fits = shape.get("fits") if isinstance(shape, dict) else None
    shape_value: Any = shape
    if fits is not None:
        pooled = next((f for f in fits if f.get("column") is None), None)
        shape_value = {
            "pooled": pooled["label"] if pooled else None,
            "by_column": {
                str(f["column"]): f["label"] for f in fits if f.get("column") is not None
            },
            "fits": fits,
        }
    return SummaryV1(
        viable_fraction=Descriptive(value=_strip_flag(desc.get("viable_fraction"))),
        upper_tool_edge=Descriptive(
            value={
                "peak": p1.get("peak"),
                "peak_interval": p1.get("peak_interval"),
                "drop_points": p1.get("drop_points"),
                "drop_interval": p1.get("drop_interval"),
                "p1": p1.get("verdict"),
            }
        ),
        boundary_width=Descriptive(value=_strip_flag(desc.get("boundary_width"))),
        recovery_time=Descriptive(
            value={
                "extra_turns": rec.get("r"),
                "interval": [rec.get("lo"), rec.get("hi")],
                "n_pairs": rec.get("n_pairs"),
                "seconds_mean": rec.get("seconds_mean"),
                "net_of_retries": rec.get("net_of_retries"),
                "p2": results["verdicts"]["p2"].get("verdict"),
                "survivorship_note": rec.get("survivorship_note"),
            }
        ),
        fault_axis_shape=Descriptive(value=shape_value),
    )


def _verdicts(results: dict[str, Any]) -> VerdictsV1:
    moves = bool(results.get("propositions_move", False))
    v = results["verdicts"]
    return VerdictsV1(
        instrument=str((results.get("instrument") or {}).get("label", "")),
        propositions_move=moves,
        p1=PropositionVerdictV1(
            verdict=v["p1"]["verdict"],
            moves=bool(v["p1"].get("moves", moves)),
            conditions=_conditions(v["p1"]),
        ),
        p2=PropositionVerdictV1(
            verdict=v["p2"]["verdict"],
            moves=bool(v["p2"].get("moves", moves)),
            conditions=_conditions(v["p2"]),
        ),
    )


def build_model_entry(
    results: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    config: SnapshotConfig,
    model: ModelConfig | None,
    repo_root: Path,
    environment: dict[str, Any],
    files: ModelFilesV1,
    jitter: dict[str, Any] | None = None,
    determinism: dict[str, Any] | None = None,
) -> ModelEntryV1:
    """One models[] entry from a model's results.json and runs.jsonl."""
    role = _role(model, results)
    costs = [r["cost_usd"] for r in runs if r.get("cost_usd") is not None]
    return ModelEntryV1(
        id=str(results["model"]),
        role=role,
        mix=str(results["mix"]),
        identity=_identity(model, results, runs, repo_root, environment, role),
        stack=_stack(role, determinism),
        grid=GridV1(
            tools_axis=list(config.sweep.tools_axis),
            fault_axis=list(config.sweep.fault_axis),
            cells=[CellV1(**c) for c in results["cells"]],
        ),
        jitter=_jitter(jitter),
        fit=_fit(results),
        summary=_summary(results),
        verdicts=_verdicts(results),
        statement=str(results["statement"]),
        spend_usd=round(sum(costs), 6) if costs else None,
        files=files,
    )


def write_snapshot_json(snapshot: SnapshotV1, path: Path) -> SnapshotV1:
    """Validate and write snapshot.json with sorted keys and LF line endings."""
    from islands_harness.stats.verdict import jsonable

    payload = jsonable(snapshot.model_dump(mode="json"))
    validated = SnapshotV1.model_validate(payload)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return validated


# --------------------------------------------------------------------------------------
# Heatmap, statement, manifest
# --------------------------------------------------------------------------------------

_SEQ_FALLBACK = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}
_SEQ_STEPS = sorted(_SEQ_FALLBACK)


def pct_text(rate: float) -> str:
    """A rate as whole percent, rounded half up. The site's page rounds the same way
    (Math.round), so a rate of exactly 92.5 percent reads 93 in the SVG, the PNG and the page;
    Python's format() would round that tie to even (92)."""
    return str(int(math.floor(rate * 100 + 0.5)))


def _seq_token(p: float) -> int:
    """Map a rate in [0, 1] to the site's 13-step sequential scale."""
    idx = min(len(_SEQ_STEPS) - 1, max(0, int(p * len(_SEQ_STEPS))))
    return _SEQ_STEPS[idx]


def write_heatmap_svg(grid: dict[str, Any], jitter: dict[str, Any] | None, path: Path) -> str:
    """Pure SVG heatmap in the mock's geometry with the site's CSS variables.

    ``grid`` is GridV1 as a dict. Each in-scope cell is a 52 px square filled with
    ``var(--seq-NNN, #fallback)`` by its rate, its rate in points and its Wilson interval as
    two text lines; the ink switches to ``on-dark`` from --seq-400 up. Rung 1 cells render
    blank with the dashed hairline outline of the mock's empty grid. A footer line reports
    the jitter regime and jitter SD when given. Returns the SVG text.
    """
    tools_axis: list[int] = grid["tools_axis"]
    fault_axis: list[float] = grid["fault_axis"]
    cells = {(c["tools"], round(c["fault"], 3)): c for c in grid["cells"]}
    x0, y0, size, pitch = 52, 46, 52, 54
    parts: list[str] = [
        '<svg class="hm" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 380" role="img" '
        'aria-label="Six by six grid, tool count across against fault rate down; each cell its success rate with a 95 percent interval.">',
        "<style>"
        ".hm text{font-family:var(--sans,system-ui,sans-serif)}"
        ".hm .ax{fill:var(--muted,#898781);font-size:12px}"
        ".hm .axt{fill:var(--ink-2,#52514e);font-size:12px;letter-spacing:.04em}"
        ".hm .cv{font-size:13px;text-anchor:middle;font-variant-numeric:tabular-nums}"
        ".hm .ci{font-size:9px;text-anchor:middle;font-variant-numeric:tabular-nums}"
        ".hm .on-light{fill:var(--ink,#0b0b0b)}.hm .on-dark{fill:var(--page,#f9f9f7)}"
        ".hm .oos rect{fill:none;stroke:var(--hairline,#e1e0d9);stroke-width:1;stroke-dasharray:4 3}"
        ".hm .lg{fill:var(--muted,#898781);font-size:10px;text-anchor:middle}"
        "</style>",
        '<text class="axt" x="213" y="16" text-anchor="middle">tools</text>',
        '<g class="ax" text-anchor="middle">'
        + "".join(
            f'<text x="{x0 + i * pitch + size // 2}" y="38">{t}</text>'
            for i, t in enumerate(tools_axis)
        )
        + "</g>",
        '<text class="axt" x="14" y="207" text-anchor="middle" transform="rotate(-90 14 207)">fault rate (percent)</text>',
        '<g class="ax" text-anchor="end">'
        + "".join(
            f'<text x="44" y="{y0 + j * pitch + 30}">{int(round(f * 100))}</text>'
            for j, f in enumerate(fault_axis)
        )
        + "</g>",
    ]
    for j, f in enumerate(fault_axis):
        for i, t in enumerate(tools_axis):
            x, y = x0 + i * pitch, y0 + j * pitch
            c = cells.get((t, round(f, 3)))
            if c is None or not c["in_scope"]:
                parts.append(
                    f'<g class="oos"><rect x="{x}" y="{y}" width="{size}" height="{size}" rx="2"/></g>'
                )
                continue
            token = _seq_token(c["p"])
            ink = "on-dark" if token >= 400 else "on-light"
            title = f"{t} tools, {int(round(f * 100))} percent faults: {c['k']}/{c['n']}, {pct_text(c['p'])} percent [{pct_text(c['lo'])}, {pct_text(c['hi'])}]"
            parts.append(
                f"<g><title>{title}</title>"
                f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="2" fill="var(--seq-{token}, {_SEQ_FALLBACK[token]})"/>'
                f'<text class="cv {ink}" x="{x + size // 2}" y="{y + 25}">{pct_text(c["p"])}</text>'
                f'<text class="ci {ink}" x="{x + size // 2}" y="{y + 40}">{pct_text(c["lo"])}–{pct_text(c["hi"])}</text></g>'
            )
    if jitter:
        sd = jitter.get("jitter_sd")
        sd_txt = f"{sd * 100:.1f} points" if isinstance(sd, (int, float)) else "n/a"
        parts.append(
            f'<text class="lg" x="213" y="376">jitter SD {sd_txt}, regime {jitter.get("regime", "n/a")}; intervals are sampling error only</text>'
        )
    parts.append("</svg>")
    svg = "\n".join(parts) + "\n"
    Path(path).write_text(svg, encoding="utf-8", newline="\n")
    return svg


def write_statement(path: Path, lines: list[str]) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_manifest(
    path: Path,
    *,
    config_hash: str,
    lock: dict[str, Any] | None,
    environment: dict[str, Any],
    identities: dict[str, Any],
    spend: dict[str, Any],
    counts: dict[str, Any],
    files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """manifest.json: everything a reviewer needs to know what ran, where, and on what.

    ``environment`` holds harness version and commit, Python and package versions, OS, GPU
    model, driver and the llama.cpp build (collect_environment); ``identities`` every
    distinct served model string and fingerprint with first and last seen dates; ``counts``
    runs by outcome, aborted_transport, transport retries and malformed calls; ``files`` the
    sha256 of the data files the snapshot was built from.
    """
    manifest = {
        "schema_version": 1,
        "config_hash": config_hash,
        "lock": lock,
        "environment": environment,
        "identities": identities,
        "spend": spend,
        "counts": counts,
        "files": files or {},
    }
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _counts(runs: list[dict[str, Any]], results: dict[str, Any]) -> dict[str, Any]:
    return {
        "runs": len(runs),
        "by_outcome": dict(sorted(Counter(str(r.get("outcome")) for r in runs).items())),
        "aborted_transport": sum(1 for r in runs if r.get("outcome") == "aborted_transport"),
        "transport_retries": sum(int(r.get("transport_retries") or 0) for r in runs),
        "malformed_calls": sum(int(r.get("malformed_calls") or 0) for r in runs),
        "torn_lines_skipped": (results.get("counts") or {}).get("torn_lines_skipped", 0),
    }


# --------------------------------------------------------------------------------------
# report: analysis outputs -> snapshot.json and its companions
# --------------------------------------------------------------------------------------

DATA_FILES = ("runs.jsonl", "results.json", "verdict.json", "cells.csv", "bootstrap/resamples.json")


def model_dirs(results_dir: Path) -> list[Path]:
    """The model directories a report covers: the directory itself when it holds a
    results.json, otherwise every immediate subdirectory that does."""
    results_dir = Path(results_dir)
    if (results_dir / "results.json").is_file():
        return [results_dir]
    return sorted(d for d in results_dir.iterdir() if d.is_dir() and (d / "results.json").is_file())


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def report(
    results_dir: Path,
    *,
    config: SnapshotConfig,
    repo_root: Path,
    kind: Literal["snapshot", "exploratory", "selftest"] = "snapshot",
    label: str | None = None,
    models: dict[str, ModelConfig] | None = None,
    environment: dict[str, Any] | None = None,
    lock: dict[str, Any] | None = None,
) -> SnapshotV1:
    """Build snapshot.json (validated), a heatmap.svg and a manifest.json per model, and the
    combined statement.txt, from the analysis outputs under ``results_dir``.

    ``models`` supplies model configs that are not in the snapshot config (the selftest's
    synthetic model); every other model is looked up in ``config``. ``lock`` is the freeze
    lock the runs were made under, recorded in each manifest (null for exploratory runs). Nothing is recomputed:
    the report restates results.json, so ``verify`` re-analyzes rather than re-reporting.
    """
    from islands_harness.stats.analyze import read_runs

    results_dir = Path(results_dir)
    dirs = model_dirs(results_dir)
    if not dirs:
        raise FileNotFoundError(f"no results.json under {results_dir}; run `islands analyze` first")
    environment = environment if environment is not None else collect_environment(repo_root)
    config_hash = sha256_bytes(canonical_bytes(config))
    prereg = load_prereg(repo_root / config.snapshot.preregistration)
    known = {m.id: m for m in config.models} | (models or {})

    entries: list[ModelEntryV1] = []
    all_dates: list[str] = []
    prereg_hashes: set[str] = set()
    snapshot_ids: set[str] = set()
    for d in dirs:
        results = json.loads((d / "results.json").read_text(encoding="utf-8"))
        runs, _ = read_runs(d / "runs.jsonl")
        model = known.get(str(results["model"]))
        jitter = _read_json(d / "jitter.json")
        determinism = _read_json(d / "determinism.json")
        rel = "" if d == results_dir else d.relative_to(results_dir).as_posix() + "/"
        grid_dict = {
            "tools_axis": list(config.sweep.tools_axis),
            "fault_axis": list(config.sweep.fault_axis),
            "cells": results["cells"],
        }
        write_heatmap_svg(grid_dict, jitter, d / "heatmap.svg")
        write_manifest(
            d / "manifest.json",
            config_hash=config_hash,
            lock=lock,
            environment=environment,
            identities=served_identities(runs),
            spend={
                "usd": round(sum(r["cost_usd"] for r in runs if r.get("cost_usd") is not None), 6),
                "price_table": str(model.prices) if model is not None and model.prices else None,
            },
            counts=_counts(runs, results),
            files={f: _sha256_file(d / f) for f in DATA_FILES if (d / f).is_file()},
        )
        entries.append(
            build_model_entry(
                results,
                runs,
                config=config,
                model=model,
                repo_root=repo_root,
                environment=environment,
                files=ModelFilesV1(
                    runs_jsonl=f"{rel}runs.jsonl",
                    results=f"{rel}results.json",
                    verdict=f"{rel}verdict.json",
                    manifest=f"{rel}manifest.json",
                    heatmap=f"{rel}heatmap.svg",
                ),
                jitter=jitter,
                determinism=determinism,
            )
        )
        all_dates.extend(request_dates(runs))
        prereg_hashes.add(str((results.get("preregistration") or {}).get("hash")))
        snapshot_ids.add(str(results["snapshot"]))
    if len(snapshot_ids) != 1:
        raise ValueError(
            f"the model directories belong to different snapshots: {sorted(snapshot_ids)}"
        )
    if len(prereg_hashes) != 1:
        raise ValueError(
            f"the model directories were analyzed under different pre-registrations: {sorted(prereg_hashes)}"
        )
    default_label = {
        "snapshot": f"Snapshot {config.snapshot.id}",
        "exploratory": f"Exploratory runs for {config.snapshot.id}: not a snapshot",
        "selftest": "Selftest: a synthetic agent on the fake provider, not a measurement",
    }[kind]
    snapshot = SnapshotV1(
        snapshot=SnapshotMetaV1(
            id=snapshot_ids.pop(),
            kind=kind,
            label=label or default_label,
            date_start=all_dates[0] if all_dates else None,
            date_end=all_dates[-1] if all_dates else None,
            preregistration=PreregRef(
                version=prereg.version, doi=prereg.deposit_doi, hash=prereg_hashes.pop()
            ),
            harness=HarnessRef(
                version=str(environment.get("harness_version") or _version("islands-harness")),
                commit=str(environment.get("commit") or "uncommitted"),
                config_hash=config_hash,
            ),
            bundle_doi=None,
        ),
        models=entries,
        links=LinksV1(
            method_note="https://islandsofstability.com/method-note",
            preregistration="https://islandsofstability.com/map#prereg",
        ),
    )
    validated = write_snapshot_json(snapshot, results_dir / "snapshot.json")
    write_statement(results_dir / "statement.txt", [e.statement for e in entries])
    return validated


# --------------------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------------------

BUNDLE_NAME = "bundle.zip"
SUMS_NAME = "SHA256SUMS"
_NOT_BUNDLED = {BUNDLE_NAME, SUMS_NAME, ".selftest", ".smoke"}
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class BundleResult:
    path: Path
    sums_path: Path
    members: int
    sha256: str  # of bundle.zip; informational, SHA256SUMS is the reference


def bundle_members(results_dir: Path) -> list[str]:
    """Every file under the results directory except the bundle itself, its SHA256SUMS and
    tool markers, as sorted POSIX paths."""
    root = Path(results_dir)
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name not in _NOT_BUNDLED and "__pycache__" not in p.parts
    )


def write_sums(results_dir: Path) -> Path:
    """SHA256SUMS in sha256sum format ("<hash>  <path>"), one line per member, sorted."""
    root = Path(results_dir)
    lines = [f"{_sha256_file(root / m)}  {m}\n" for m in bundle_members(root)]
    path = root / SUMS_NAME
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return path


def bundle(results_dir: Path) -> BundleResult:
    """Zip the results directory (runs, transcripts, analysis, bootstrap indices, report
    files) into bundle.zip beside a SHA256SUMS file, for Zenodo. Members are stored in sorted
    order with a fixed timestamp, fixed attributes and the SHA256SUMS file last, so the same
    inputs give the same member bytes everywhere."""
    root = Path(results_dir)
    sums = write_sums(root)
    members = [*bundle_members(root), SUMS_NAME]
    out = root / BUNDLE_NAME
    with zipfile.ZipFile(out, "w") as zf:
        for name in members:
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            zf.writestr(info, (root / name).read_bytes(), compresslevel=9)
    return BundleResult(out, sums, len(members), _sha256_file(out))


def check_sums(results_dir: Path) -> dict[str, list[str]]:
    """Recompute every hash SHA256SUMS lists. Returns {"mismatched", "missing", "unlisted"}:
    the first two are failures; unlisted files (added after the bundle) are reported only."""
    root = Path(results_dir)
    listed: dict[str, str] = {}
    for line in (root / SUMS_NAME).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, name = line.partition("  ")
            listed[name] = digest
    mismatched = [
        n for n, h in listed.items() if (root / n).is_file() and _sha256_file(root / n) != h
    ]
    missing = [n for n in listed if not (root / n).is_file()]
    unlisted = [m for m in bundle_members(root) if m not in listed]
    return {"mismatched": sorted(mismatched), "missing": sorted(missing), "unlisted": unlisted}


# --------------------------------------------------------------------------------------
# Optional PNG
# --------------------------------------------------------------------------------------


def write_heatmap_png(grid: dict[str, Any], png_path: Path) -> Path:
    """Optional PNG of the heatmap through the ``plots`` extra (matplotlib), drawn from the
    grid data. The site never needs it; the SVG is the published figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "PNG export needs the optional plots extra: uv sync --extra plots"
        ) from exc
    tools_axis, fault_axis = grid["tools_axis"], grid["fault_axis"]
    cells = {(c["tools"], round(c["fault"], 3)): c for c in grid["cells"]}
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=200)
    for j, f in enumerate(fault_axis):
        for i, t in enumerate(tools_axis):
            c = cells.get((t, round(f, 3)))
            if c is None or not c["in_scope"]:
                ax.add_patch(plt.Rectangle((i, j), 0.96, 0.96, fill=False, ls="--", lw=0.5))
                continue
            colour = _SEQ_FALLBACK[_seq_token(c["p"])]
            ax.add_patch(plt.Rectangle((i, j), 0.96, 0.96, color=colour))
            ink = "white" if _seq_token(c["p"]) >= 400 else "black"
            ax.text(
                i + 0.48,
                j + 0.45,
                pct_text(c["p"]),
                ha="center",
                va="center",
                fontsize=8,
                color=ink,
            )
    ax.set_xlim(0, len(tools_axis))
    ax.set_ylim(len(fault_axis), 0)
    ax.set_xticks([i + 0.48 for i in range(len(tools_axis))], [str(t) for t in tools_axis])
    ax.set_yticks(
        [j + 0.48 for j in range(len(fault_axis))], [f"{int(round(f * 100))}" for f in fault_axis]
    )
    ax.set_xlabel("tools")
    ax.set_ylabel("fault rate (percent)")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return Path(png_path)


# --------------------------------------------------------------------------------------
# The pre-registration deposit
# --------------------------------------------------------------------------------------


REDACTIONS_NAME = "REDACTIONS.txt"

# How a path separator appears in text. Inside a JSON string a backslash is always escaped,
# so there a lone backslash starts an escape such as \n and is never a separator.
_SEP_JSON = r"(?:\\\\|/)"
_SEP_TEXT = r"(?:\\\\|\\|/)"


def _path_regex(path: Path, sep: str) -> str:
    """``path`` as a regex: the drive, if any, then each part, joined by ``sep``."""
    return re.escape(path.drive) + sep + sep.join(re.escape(part) for part in path.parts[1:])


def _name_ends(sep: str) -> str:
    """Lookahead for the end of a path name: a separator, a quote, whitespace, a closing
    bracket, a comma or semicolon, or the end of the text."""
    return rf"(?={sep}|[\"'\s)\],;]|$)"


def shorten_local_paths(text: str, repo_root: Path, home: Path, *, json_text: bool) -> str:
    """``text`` with this machine's absolute paths shortened: the repository root becomes
    "." (a path inside it keeps its separators after the dot), and any other path under the
    home directory starts with "~". Case-insensitive, as Windows paths are, in either
    separator spelling; a JSON text stays valid JSON."""
    sep = _SEP_JSON if json_text else _SEP_TEXT
    root = _path_regex(Path(repo_root).resolve(), sep)
    user = _path_regex(Path(home).resolve(), sep)
    text = re.sub(root + _name_ends(sep), ".", text, flags=re.IGNORECASE)
    return re.sub(user + _name_ends(sep), "~", text, flags=re.IGNORECASE)


def _names_local_user(text: str, home: Path) -> str | None:
    """The first path in ``text`` that names the home directory's user, in full or as a
    Windows short name (JOHNSM~1 for johnsmith), in any separator spelling."""
    name = Path(home).resolve().name
    if not name:
        return None
    forms = [re.escape(name) + _name_ends(_SEP_TEXT)]
    stem = re.sub(r"\W", "", name)[:6]
    if stem and stem != name:
        forms.append(re.escape(stem) + r"~\d")
    found = re.search(_SEP_TEXT + "(?:" + "|".join(forms) + ")", text, flags=re.IGNORECASE)
    return None if found is None else found.group(0)


def _redactions_note(files: list[str]) -> str:
    listed = "".join(f"  {f}\n" for f in sorted(files))
    return (
        "Local paths shortened in this deposit\n\n"
        "The files below are copies of the harness repository's files with one change:\n"
        "absolute paths on the author's machine were shortened. The repository root is\n"
        "written as . and any other path under the author's home directory begins with ~.\n"
        "Nothing else differs; SHA256SUMS hashes the copies as deposited.\n\n" + listed
    )


def deposit_files(config: SnapshotConfig, repo_root: Path, config_path: Path) -> list[Path]:
    """What the dated pre-registration deposit holds, dated before any full cell runs: the
    rules (r4.yaml and its prose mirror), the snapshot config and its freeze lock, the method
    note, each model's calibration and determinism-gate record, the joint calibration
    recommendation, and any hosted spend forecast. Paths relative to the repository root."""
    root = Path(repo_root)
    outputs = root / config.snapshot.outputs
    lock = config_path.parent / f"{config_path.stem}.lock.json"
    files = [
        root / config.snapshot.preregistration,
        root / "PREREGISTRATION.md",
        config_path,
        lock,
        root / "docs" / "method-note.md",
        outputs / "calibration-recommendation.json",
    ]
    for m in config.models:
        files += [
            outputs / m.id / "calibration" / "calibration.json",
            outputs / m.id / "determinism.json",
            outputs / m.id / "forecast.json",
        ]
    return [f for f in files if f.is_file()]


def prepare_deposit(
    config: SnapshotConfig,
    repo_root: Path,
    config_path: Path,
    out_dir: Path,
    *,
    home: Path | None = None,
) -> BundleResult:
    """Copy the deposit files under ``out_dir`` at their repository paths and bundle them
    (SHA256SUMS plus bundle.zip). Refuses without the freeze lock: the deposit exists to
    date the frozen rules. Nothing is uploaded; the Zenodo upload is the author's step.

    The copies shorten this machine's absolute paths, which the gate records hold for the
    server binary and the weights (``shorten_local_paths``; ``home`` defaults to the user's
    home directory). REDACTIONS.txt lists the files that changed, and nothing is written
    while any copy still names the local user in a path."""
    import shutil

    lock = config_path.parent / f"{config_path.stem}.lock.json"
    if not lock.is_file():
        raise FileNotFoundError(
            f"no freeze lock at {lock}; the deposit comes after `islands freeze`"
        )
    root = Path(repo_root).resolve()
    home = Path.home() if home is None else Path(home)
    copies: dict[str, str] = {}
    shortened: list[str] = []
    for src in deposit_files(config, repo_root, config_path):
        rel = Path(src).resolve().relative_to(root).as_posix()
        text = Path(src).read_bytes().decode("utf-8")
        is_json = Path(src).suffix == ".json"
        copy = shorten_local_paths(text, root, home, json_text=is_json)
        if is_json:
            json.loads(copy)  # still valid JSON
        mention = _names_local_user(copy, home)
        if mention is not None:
            raise ValueError(f"{rel} still names the local user in a path ({mention})")
        if copy != text:
            shortened.append(rel)
        copies[rel] = copy
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    for rel, copy in copies.items():
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(copy.encode("utf-8"))
    if shortened:
        (out / REDACTIONS_NAME).write_text(
            _redactions_note(shortened), encoding="utf-8", newline="\n"
        )
    return bundle(out)
