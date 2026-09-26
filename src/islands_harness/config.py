"""Frozen configuration models, canonical hashing, freeze and lock checking.

Two files define a snapshot: the config (what to run) and the pre-registration (how to
decide). Both load into frozen pydantic models with ``extra="forbid"``, so an unknown or
misspelled key is an error rather than a silently ignored knob. Nothing outside these two
files changes a measured quantity (ARCHITECTURE.md Section 5).

Why each input is hashed by ``freeze`` (Section 8 of the design, D8):

    config              every knob that shapes a run: limits, sampling, axes, fault kinds
    pre-registration    every decision rule; ``analyze`` refuses a changed hash
    system prompt       byte for byte; the model sees these bytes
    tool schemas        per rung, as sent to the model; a description edit changes behaviour
    dataset             documents and gold, per file; a regenerated dataset is a new snapshot
    grader source       the score is a function of this file
    loop source         the loop is the instrument; any edit reopens the audit
    faults source       the fault kinds and their frozen result strings
    providers source    what leaves the machine and what comes back

Hashing is canonical so a Windows checkout and a Linux checkout agree: text files are read
as UTF-8 with CRLF normalized to LF before hashing; structured data is serialized with sorted
keys and no whitespace. The hashes are recorded in ``configs/<snapshot>.lock.json`` and
checked by every run command after the freeze.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------------------


class _Frozen(BaseModel):
    """Immutable, strict base: unknown keys are errors, instances are hashable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------
# Snapshot config (configs/snapshot-1.yaml)
# --------------------------------------------------------------------------------------


class SnapshotMeta(_Frozen):
    id: str
    root_seed: int
    preregistration: Path
    harness_tag: str
    outputs: Path


class TaskConfig(_Frozen):
    dir: Path
    system_prompt: Path
    dataset: Path
    grader: Path
    required_tools: list[str] = Field(min_length=2, max_length=2)
    mixes: dict[str, list[str]]
    rung1: list[str] = Field(min_length=1, max_length=1)


class AgentConfig(_Frozen):
    max_turns: int = Field(ge=1)
    max_tool_calls: int = Field(ge=1)
    token_limit: int = Field(ge=1)
    time_limit_s: float = Field(gt=0)
    parallel_tool_calls: Literal["allow", "disallow"]
    tool_choice: Literal["auto"]  # never forced (D11)
    strict_schemas: Literal[False]  # never strict; argument validation is the model's job
    on_stop_without_submit: Literal["end"]
    on_malformed_arguments: Literal["catalog_error"]


class SweepConfig(_Frozen):
    tools_axis: list[int]
    in_scope_tools: list[int]
    fault_axis: list[float]
    mixes_to_run: list[str]
    order: Literal["round_robin", "sequential"]
    rung1_runs_per_cell: int = Field(ge=0)


class GarbleConfig(_Frozen):
    char_fraction: float = Field(gt=0, lt=1)
    truncate: bool


class FaultConfig(_Frozen):
    kinds: list[Literal["timeout", "garbled_payload", "error_response", "empty_result"]]
    weights: list[float]
    applies_to: Literal["all_tools", "exempt_submit"]  # exempt_submit is a client option only
    nested_across_rates: Literal[True]
    simulate_delay_s: float = Field(ge=0)
    garble: GarbleConfig


class ProbeConfig(_Frozen):
    runs_per_point: int = Field(ge=1)
    tools_points: list[int]
    fault_points: list[float]


class CellRef(_Frozen):
    tools: int
    fault: float


class CalibrationConfig(_Frozen):
    anchor_cell: CellRef
    target_band: tuple[float, float]
    pilot_runs: int = Field(ge=1)
    difficulty_levels: list[int]
    tau_candidates: list[float]
    max_malformed_call_rate: float = Field(ge=0, le=1)


class LocalFloorConfig(_Frozen):
    cell: CellRef
    fixed_seed_reruns: int = Field(ge=1)
    # Documents per fixed-seed rerun; None means the cell's full document set. On a stack the
    # gate has verified deterministic, fixed-seed reruns are a check, not a measurement, so a
    # block of documents suffices (D12).
    fixed_seed_documents: int | None = Field(default=None, ge=1)
    varied_seed_reruns: int = Field(ge=1)


class HostedFloorConfig(_Frozen):
    cells: list[CellRef]
    reruns: int = Field(ge=1)
    documents_per_rerun: int = Field(ge=1)


class NoiseFloorConfig(_Frozen):
    local: LocalFloorConfig
    hosted: HostedFloorConfig


class GateConfig(_Frozen):
    specs: int = Field(ge=1)
    replays_per_regime: int = Field(ge=1)
    filler_requests: int = Field(ge=0)


class SamplingConfig(_Frozen):
    """What the harness asks of the sampler; providers/base.CAPABILITIES says who accepts what."""

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = Field(ge=1)
    seed: Literal["per_run"] | int | None = None
    thinking: Literal["adaptive", "disabled"] | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class ExpectConfig(_Frozen):
    """What ``doctor`` and ``serve`` verify about a local server before a run (D9).

    For llama.cpp every field is checked against the running server's ``GET /props``
    (``build_info``, ``model_path``, ``total_slots``, ``n_ctx``, chat format) and against the
    weights file on disk (sha256 recomputed). A mismatch blocks the run.
    """

    server: Literal["llama.cpp", "other"]
    build: str  # llama.cpp build tag, for example "b11191"
    build_commit: str  # full commit hash of that build
    weights_file: str  # file name inside models/weights/
    weights_sha256: str
    quantization: str  # as published, for example "MXFP4" or "Q5_K_M"; recorded, not converted
    ctx_size: int = Field(ge=1024)
    parallel_slots: Literal[1]  # one slot: batch composition cannot vary (D12)
    cache_prompt: Literal[False]  # llama.cpp documents prompt caching as a source of nondeterminism
    flash_attn: Literal["on", "off"]
    kv_cache_type: Literal["f16"]
    batch_size: int = Field(ge=1)
    ubatch_size: int = Field(ge=1)
    chat_format: str  # what /props reports, for example "GPT-OSS"; TODO(M0): confirm per model
    deterministic: Literal["verify_with_gate"]  # never asserted; the gate labels the stack


class ModelConfig(_Frozen):
    id: str
    provider: Literal["openai_compat", "anthropic"]
    model: str
    allowed_hosts: list[str] = Field(min_length=1)
    documents: int = Field(ge=1)
    epochs: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    sampling: SamplingConfig
    # openai_compat
    base_url: str | None = None
    api_key_env: str | None = None
    lock: Path | None = None
    extra_body: dict[str, Any] | None = None
    expect: ExpectConfig | None = None
    # anthropic
    transport: Literal["anthropic", "bedrock", "vertex", "foundry"] | None = None
    # region, project_id, resource or base_url for the cloud transports; kept here rather than
    # in environment variables so they are part of the hashed config
    transport_options: dict[str, str] | None = None
    prompt_caching: bool = False
    fallbacks: Literal["none"] = "none"  # never sent (D11)
    prices: Path | None = None
    spend_cap_usd: float | None = Field(default=None, gt=0)


class SnapshotConfig(_Frozen):
    schema_version: Literal[1]
    snapshot: SnapshotMeta
    task: TaskConfig
    agent: AgentConfig
    sweep: SweepConfig
    faults: FaultConfig
    probe: ProbeConfig
    calibration: CalibrationConfig
    noise_floor: NoiseFloorConfig
    determinism_gate: GateConfig
    models: list[ModelConfig] = Field(min_length=1)

    def model_by_id(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"no model with id {model_id!r} in config")


# --------------------------------------------------------------------------------------
# Pre-registration (configs/preregistration/r4.yaml)
# --------------------------------------------------------------------------------------


class SuccessRule(_Frozen):
    rule: Literal["score_ge_tau"]
    tau: float = Field(gt=0, le=1)
    chosen_by: Literal["calibration"]
    band: tuple[float, float]
    difficulty: int | None
    max_malformed_call_rate: float


class IntervalRules(_Frozen):
    cell: Literal["wilson"]
    difference: Literal["newcombe"]
    level: float
    display_rule: str


class SeparationRule(_Frozen):
    nonconvergence: bool
    max_abs_coefficient: float
    saturated_column_alone: bool


class PeakRule(_Frozen):
    formula: str
    range: tuple[float, float]


class DropRule(_Frozen):
    row: float
    comparison: str
    peak_column: str
    interval: Literal["newcombe"]
    sesoi_points: float


class FaultCapCheck(_Frozen):
    rule: str
    on_failure: Literal["blocked"]


class BootstrapRule(_Frozen):
    draws: int = Field(ge=1)
    unit: Literal["document"]
    interval: Literal["percentile"]
    scope: str
    keep_nonnegative_b2_resamples: bool = True
    # P1 only. The implementation supports exactly these readings (stats/fit.py), so the
    # Literal makes the pre-registration and the code agree at load time.
    refit: Literal["ladder_per_resample"] | None = None
    no_peak_resamples: Literal["signed_infinity_by_slope"] | None = None
    failed_resamples: Literal["counted_as_no_peak"] | None = None


class P1Scope(_Frozen):
    mix: str
    tools: list[int]
    faults: Literal["all"]
    outcomes_excluded: list[str]


class P1Rules(_Frozen):
    proposition: str
    model: str
    covariates: dict[str, str]
    family: Literal["binomial"]
    link: Literal["logit"]
    unit: Literal["run"]
    scope: P1Scope
    covariance: Literal["cluster_by_document"]
    ladder: list[Literal["mle_cluster", "firth_bootstrap", "not_estimable"]]
    separation_rule: SeparationRule
    precedence: list[
        Literal["not_estimable", "blocked", "survives", "restricted", "below_resolution"]
    ]
    curvature: str
    peak: PeakRule
    drop: DropRule
    fault_cap_check: FaultCapCheck
    bootstrap: BootstrapRule


class P2Rules(_Frozen):
    proposition: str
    population: str
    control: str
    unit: Literal["turns"]
    statistic: str
    rule: str
    sesoi_turns: float
    secondary: str
    seconds: Literal["descriptive"]
    bootstrap: BootstrapRule
    pairing_note: str
    no_pairs: Literal["not_estimable"]  # no recovered faulted run with a control


class BelowResolutionRules(_Frozen):
    rule: Literal["sesoi_equivalence"]
    p1: str
    p2: str
    instrument: str
    # every P1 case matching no named branch (stats/verdict.py records it as residual_branch)
    residual: Literal["below_resolution"]


class Rung1Rules(_Frozen):
    tools: Literal[1]
    excluded_from_all_statistics: Literal[True]
    runs_per_cell: int
    reports: str


class LocalFloorRules(_Frozen):
    cell: CellRef
    fixed_seed_reruns: int
    fixed_seed_documents: int | None = None  # None means the full cell (departure 7)
    varied_seed_reruns: int
    regime: str
    reports: list[str]


class HostedFloorRules(_Frozen):
    cells: list[CellRef]
    reruns: int
    block_documents: int
    reading: Literal["block", "full_cell"]


class NoiseFloorRules(_Frozen):
    local: LocalFloorRules
    hosted: HostedFloorRules
    reported_beside_intervals_never_inside: Literal[True]


class PreRegistration(_Frozen):
    version: str
    supersedes: str | None
    frozen_on: str | None
    deposit_doi: str | None
    alpha: float
    success: SuccessRule
    intervals: IntervalRules
    p1: P1Rules
    p2: P2Rules
    below_resolution: BelowResolutionRules
    rung1: Rung1Rules
    noise_floor: NoiseFloorRules
    descriptive: list[str]
    departures_from_r3: list[str]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8", newline=None) as f:
        return yaml.safe_load(f)


def load_config(path: Path | str) -> SnapshotConfig:
    """Load and validate a snapshot config. Paths inside stay relative to the repo root."""
    return SnapshotConfig.model_validate(_read_yaml(Path(path)))


def load_models(path: Path | str) -> list[ModelConfig]:
    """Model entries from a file holding only a ``models`` list, such as
    configs/hosted-models.yaml (hosted definitions kept outside snapshot 1)."""
    return [ModelConfig.model_validate(m) for m in _read_yaml(Path(path))["models"]]


def load_prereg(path: Path | str) -> PreRegistration:
    """Load and validate a pre-registration file."""
    return PreRegistration.model_validate(_read_yaml(Path(path)))


# --------------------------------------------------------------------------------------
# Canonical hashing
# --------------------------------------------------------------------------------------


def canonical_bytes(obj: Any) -> bytes:
    """Canonical UTF-8 bytes of a JSON-compatible object: sorted keys, no whitespace, LF only.

    Pydantic models are dumped with ``mode="json"`` first so Paths and tuples serialize the
    same way on every platform (Paths as POSIX strings).
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    obj = _posixify(obj)
    text = json.dumps(
        obj, separators=(",", ":"), sort_keys=True, ensure_ascii=True, allow_nan=False
    )
    return text.replace("\r\n", "\n").encode("utf-8")


def _posixify(obj: Any) -> Any:
    """Render Path values as forward-slash strings so Windows and Linux hash identically."""
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {k: _posixify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_posixify(v) for v in obj]
    return obj


DEPOSIT_METADATA = ("frozen_on", "deposit_doi")


def prereg_hash(prereg: PreRegistration) -> str:
    """The pre-registration's hash: every rule, but not its deposit metadata.

    ``frozen_on`` and ``deposit_doi`` describe the deposit rather than decide anything, and
    both can only be known at or after the freeze (the DOI is reserved on the deposit's
    draft). Leaving them out lets them be filled in without breaking the lock, while any
    change to a rule, threshold or category still changes the hash. Every place that records
    or checks the pre-registration's identity uses this function.
    """
    data = prereg.model_dump(mode="json")
    for key in DEPOSIT_METADATA:
        data.pop(key, None)
    return sha256_bytes(canonical_bytes(data))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, normalize_newlines: bool = True) -> str:
    """sha256 of a file. Text files are hashed with CRLF normalized to LF.

    With ``normalize_newlines=True`` the file is decoded as UTF-8 (strict) and CRLF becomes LF
    before hashing, so a checkout with ``core.autocrlf`` on Windows hashes the same as a Linux
    checkout. Pass ``False`` for binary files (model weights, images).
    """
    p = Path(path)
    data = p.read_bytes()
    if normalize_newlines:
        data = data.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    return sha256_bytes(data)


# --------------------------------------------------------------------------------------
# Freeze and lock
# --------------------------------------------------------------------------------------


class LockMismatch(Exception):
    """A hashed input differs from the lock written by ``islands freeze``."""


class SnapshotLock(_Frozen):
    """What ``islands freeze`` writes to ``configs/<snapshot>.lock.json``.

    Every field is a hex sha256 over canonical bytes, except ``tool_schemas`` which maps
    ``"<mix>:<k>"`` to the hash of the exact tool list sent to the model at that rung, and
    ``dataset`` which maps relative file path to hash.
    """

    schema_version: Literal[1] = 1
    frozen_on: str
    harness_tag: str
    git_commit: str
    config: str
    preregistration: str
    system_prompt: str
    tool_schemas: dict[str, str]
    dataset: dict[str, str]
    grader_source: str
    loop_source: str
    faults_source: str
    providers_source: dict[str, str]
    tools_source: str  # the harness's execute path (islands_harness/tools.py)
    task_tools_source: str  # the task's tool implementations (tasks/<task>/tools.py)


_PACKAGE_DIR = Path(__file__).resolve().parent


def freeze(
    config: SnapshotConfig, repo_root: Path, *, frozen_on: str, git_commit: str
) -> SnapshotLock:
    """Hash every fixed input and return the lock. The CLI writes it and refuses a dirty tree.

    Config and pre-registration are hashed as parsed models (canonical_bytes), so comments
    and formatting do not move the hash but every value does. Source files are hashed with
    CRLF normalized. Tool schemas are hashed per rung and mix exactly as sent. Dataset hashes
    come from the dataset's LOCK.json after verifying it; a dataset that has not been
    generated yet hashes as empty, and the CLI's ``freeze`` refuses that case.
    """
    from islands_harness import dataset as ds
    from islands_harness.tools import Registry

    task_dir = repo_root / config.task.dir
    registry = Registry.load(
        task_dir / "tools.py",
        required=config.task.required_tools,
        mixes=config.task.mixes,
        rung1=config.task.rung1,
    )
    schemas = {
        f"{mix}:{k}": registry.schema_hash(registry.for_rung(mix, k))
        for mix in sorted(config.task.mixes)
        for k in config.sweep.tools_axis
    }
    dataset_dir = task_dir / config.task.dataset
    dataset_files: dict[str, str] = {}
    if (dataset_dir / ds.DATASET_LOCK_NAME).exists():
        dataset_files = dict(ds.verify_lock(dataset_dir)["files"])
    providers = sorted(
        p for p in (_PACKAGE_DIR / "providers").glob("*.py") if p.name != "__init__.py"
    )
    return SnapshotLock(
        frozen_on=frozen_on,
        harness_tag=config.snapshot.harness_tag,
        git_commit=git_commit,
        config=sha256_bytes(canonical_bytes(config)),
        preregistration=prereg_hash(load_prereg(repo_root / config.snapshot.preregistration)),
        system_prompt=sha256_file(task_dir / config.task.system_prompt),
        tool_schemas=schemas,
        dataset=dataset_files,
        grader_source=sha256_file(task_dir / config.task.grader),
        loop_source=sha256_file(_PACKAGE_DIR / "loop.py"),
        faults_source=sha256_file(_PACKAGE_DIR / "faults.py"),
        providers_source={p.name: sha256_file(p) for p in providers},
        tools_source=sha256_file(_PACKAGE_DIR / "tools.py"),
        task_tools_source=sha256_file(task_dir / "tools.py"),
    )


def check_lock(config: SnapshotConfig, lock: SnapshotLock, repo_root: Path) -> None:
    """Recompute every hash and raise LockMismatch naming the first input that differs.

    Called by every run command after the freeze and by ``analyze`` (for the pre-registration
    and the config). ``frozen_on`` and ``git_commit`` are not recomputed.
    """
    fresh = freeze(config, repo_root, frozen_on=lock.frozen_on, git_commit=lock.git_commit)
    for name in SnapshotLock.model_fields:
        if name in ("frozen_on", "git_commit"):
            continue
        if getattr(fresh, name) != getattr(lock, name):
            raise LockMismatch(f"{name} differs from the lock written on {lock.frozen_on}")


def write_lock(lock: SnapshotLock, path: Path) -> None:
    path.write_bytes(canonical_bytes(lock) + b"\n")


def read_lock(path: Path) -> SnapshotLock:
    return SnapshotLock.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
