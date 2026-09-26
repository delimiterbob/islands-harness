"""Determinism gate, noise floor protocols, probe, spend estimate, doctor, selftest and replay.

Everything that makes a statement about the instrument rather than about the agent lives
here (ARCHITECTURE.md Section 8). None of it is assumed: the gate measures whether the stack
reproduces a transcript, the floor measures the residual run-to-run jitter in the sweep's own
regime, the probe measures tokens per run before money is spent, and replay re-executes one
run to show a reviewer the diff.

Canonical transcript hash: sha256 over the canonical JSON of the sequence of (role, text,
tool calls with name and argument text, tool result contents) for every message, with
provider identity fields and timings excluded, so two executions that produced the same
model-visible bytes hash the same.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from islands_harness.config import ModelConfig, SnapshotConfig, load_prereg
from islands_harness.providers.base import ChatProvider, Message


def canonical_rows(transcript: list[Message]) -> list[Any]:
    """The model-visible content of each message, the unit the transcript hash covers."""
    rows: list[Any] = []
    for m in transcript:
        rows.append(
            [
                m.role,
                m.content,
                [[c.name, c.arguments_text] for c in (m.tool_calls or [])],
                [[r.name, r.content] for r in (m.tool_results or [])],
                # Reasoning text (llama-server's reasoning_content) is part of identity: a stack
                # whose reasoning drifts while the actions happen to agree is not deterministic.
                [
                    b.get("text")
                    for b in (m.raw_blocks or [])
                    if b.get("type") == "reasoning_content"
                ],
            ]
        )
        # Tool-call ids are excluded on purpose: llama-server generates them at random.
    return rows


def canonical_transcript_hash(transcript: list[Message]) -> str:
    payload = json.dumps(
        canonical_rows(transcript), separators=(",", ":"), sort_keys=True, ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Determinism gate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    passed: bool
    regimes: tuple[str, ...]  # ("serial", "concurrent_shuffled", "concurrent_filler")
    specs: int
    replays_per_regime: int
    hashes: dict[str, list[str]]  # run_id -> the 15 transcript hashes, regime-major
    disagreeing_specs: list[str]
    variant: str  # "default" | "flash_attn_off" | another named server-flag change
    label: str  # "deterministic_verified" | "non_deterministic"
    model_id: str = ""
    executions: int = 0
    filler_requests: int = 0
    filler_failures: int = 0
    seconds: float = 0.0
    outcomes: dict[str, int] = field(default_factory=dict)
    divergences: list[dict[str, Any]] = field(default_factory=list)
    server: dict[str, Any] = field(default_factory=dict)  # /props and command-line evidence
    gpu: dict[str, Any] | None = None
    config_hash: str = ""
    run_on: str = ""


GATE_REGIMES = ("serial", "concurrent_shuffled", "concurrent_filler")
GATE_CONCURRENCY = 4  # "four at a time" (Section 8): they queue at the single slot
_FILLER_WORDS = (
    "river stone lamp orchard ledger harbor violet copper meadow signal cedar lantern "
    "compass garnet willow quarry ember tide canyon falcon marble thistle beacon prairie "
    "anchor glacier saffron timber cobalt heron summit basalt pepper juniper atlas kettle "
    "mosaic nectar pylon quartz ripple sparrow tundra umber vessel walnut yarrow zephyr "
    "almond bramble cinder delta fennel grove hollow indigo jasper kelp linen mantle"
).split()


def _filler_messages(
    config: SnapshotConfig, model: ModelConfig, variant: str, i: int
) -> list[Message]:
    """An unrelated short prompt of hashed length (5 to 400 words), so the requests queued
    around the gate's runs vary in size and cross micro-batch boundaries."""
    from islands_harness.rng import unit

    root = config.snapshot.root_seed
    n = 5 + int(unit("filler", root, model.id, variant, i, "length") * 396)
    words = [
        _FILLER_WORDS[int(unit("filler", root, model.id, variant, i, k) * len(_FILLER_WORDS))]
        for k in range(n)
    ]
    return [Message("user", "Summarize in one sentence: " + " ".join(words))]


async def _gate_async(
    config: SnapshotConfig,
    model: ModelConfig,
    provider: ChatProvider | None,
    *,
    repo_root: Path,
    out_dir: Path,
    variant: str,
    progress: Any = None,
) -> tuple[GateResult, list[dict[str, Any]]]:
    import time

    from islands_harness.providers.base import Sampling
    from islands_harness.providers.factory import build_provider, sampling_for
    from islands_harness.rng import seed64, unit
    from islands_harness.runner import execute_spec, load_task, message_to_json
    from islands_harness.specs import gate_specs

    gate = config.determinism_gate
    task = load_task(config, repo_root)
    specs = gate_specs(config, model, sorted(task.documents), config.sweep.mixes_to_run[0])
    own = provider is None
    provider = provider or build_provider(model, config.agent)
    base = Path(out_dir) / "gate" / variant
    if base.exists():
        shutil.rmtree(base)  # a rerun of the same variant replaces its own records only
    hashes: dict[str, list[str]] = {s.run_id: [] for s in specs}
    first_rows: dict[tuple[str, str], list[Any]] = {}
    executions: list[dict[str, Any]] = []
    filler_failures = 0
    started = time.monotonic()

    async def run(spec: Any, regime: str, replay: int) -> tuple[str, str]:
        t0 = time.monotonic()
        record, transcript = await execute_spec(spec, provider, task, config=config, model=model)
        digest = canonical_transcript_hash(transcript)
        first_rows.setdefault((spec.run_id, digest), canonical_rows(transcript))
        path = base / regime / f"r{replay}" / f"{spec.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"run_id": spec.run_id, "messages": [message_to_json(m) for m in transcript]},
                indent=1,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        executions.append(
            {
                "run_id": spec.run_id,
                "regime": regime,
                "replay": replay,
                "hash": digest,
                "outcome": record.outcome.value,
                "turns": record.turns,
                "fingerprints": list(record.fingerprints),
                "seconds": round(time.monotonic() - t0, 3),
            }
        )
        return spec.run_id, digest

    async def filler(i: int) -> None:
        nonlocal filler_failures
        base_sampling = sampling_for(
            model, seed64("filler", config.snapshot.root_seed, model.id, variant, i)
        )
        sampling = Sampling(
            temperature=base_sampling.temperature,
            top_p=base_sampling.top_p,
            max_tokens=24,
            seed=base_sampling.seed,
            thinking=base_sampling.thinking,
            effort=base_sampling.effort,
        )
        try:
            await provider.complete(_filler_messages(config, model, variant, i), [], sampling)
        except Exception:  # noqa: BLE001 - a filler only occupies the queue; its failure is counted
            filler_failures += 1

    per_replay = [
        gate.filler_requests // gate.replays_per_regime
        + (1 if r < gate.filler_requests % gate.replays_per_regime else 0)
        for r in range(gate.replays_per_regime)
    ]
    try:
        for regime in GATE_REGIMES:
            for replay in range(gate.replays_per_regime):
                if regime == "serial":
                    results = [await run(s, regime, replay) for s in specs]
                else:
                    order = sorted(
                        specs,
                        key=lambda s: unit(
                            "gate-order", model.id, variant, regime, replay, s.run_id
                        ),
                    )
                    semaphore = asyncio.Semaphore(GATE_CONCURRENCY)

                    async def guarded(
                        s: Any,
                        regime: str = regime,
                        replay: int = replay,
                        semaphore: asyncio.Semaphore = semaphore,
                    ) -> tuple[str, str]:
                        async with semaphore:
                            return await run(s, regime, replay)

                    jobs = [guarded(s) for s in order]
                    if regime == "concurrent_filler":
                        offset = sum(per_replay[:replay])
                        filler_semaphore = asyncio.Semaphore(GATE_CONCURRENCY)

                        async def guarded_filler(
                            i: int, filler_semaphore: asyncio.Semaphore = filler_semaphore
                        ) -> None:
                            async with filler_semaphore:
                                await filler(i)

                        jobs += [guarded_filler(offset + i) for i in range(per_replay[replay])]
                    results = [r for r in await asyncio.gather(*jobs) if r is not None]
                for run_id, digest in results:
                    hashes[run_id].append(digest)
                if progress is not None:
                    progress(f"{regime} replay {replay + 1}/{gate.replays_per_regime} done")
    finally:
        if own:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()

    disagreeing = sorted(r for r, hs in hashes.items() if len(set(hs)) > 1)
    divergences = []
    for run_id in disagreeing:
        seen: dict[str, list[str]] = {}
        for e in executions:
            if e["run_id"] == run_id:
                seen.setdefault(e["hash"], []).append(f"{e['regime']}/r{e['replay']}")
        variants = list(seen)
        a, b = first_rows[(run_id, variants[0])], first_rows[(run_id, variants[1])]
        first = next(
            (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
        )
        divergences.append(
            {
                "run_id": run_id,
                "distinct_transcripts": len(variants),
                "where": {h[:12]: places for h, places in seen.items()},
                "first_divergent_message": first,
            }
        )
    passed = not disagreeing
    result = GateResult(
        passed=passed,
        regimes=GATE_REGIMES,
        specs=len(specs),
        replays_per_regime=gate.replays_per_regime,
        hashes=hashes,
        disagreeing_specs=disagreeing,
        variant=variant,
        label="deterministic_verified" if passed else "non_deterministic",
        model_id=model.id,
        executions=len(executions),
        filler_requests=gate.filler_requests,
        filler_failures=filler_failures,
        seconds=round(time.monotonic() - started, 1),
        outcomes=dict(
            sorted(__import__("collections").Counter(e["outcome"] for e in executions).items())
        ),
        divergences=divergences,
    )
    return result, executions


def verify_determinism(
    config: SnapshotConfig,
    model: ModelConfig,
    provider: ChatProvider | None = None,
    *,
    repo_root: Path | None = None,
    out_dir: Path | None = None,
    variant: str = "default",
    server: dict[str, Any] | None = None,
    gpu: dict[str, Any] | None = None,
    progress: Any = None,
) -> GateResult:
    """The three-regime gate.

    Take ``determinism_gate.specs`` fixed specs from the anchor cell (specs.gate_specs) and
    execute each ``replays_per_regime`` times in three regimes: serially; submitted four at a
    time in hash-shuffled order; and four at a time with ``filler_requests`` unrelated
    requests in flight. On a one-slot llama.cpp server the concurrent regimes queue, so they
    test that queueing and submission order leave every transcript unchanged (short prompts drawn by rng scope
    "filler"). Hash every canonical transcript; pass when all hashes agree for every spec.
    Writes determinism.json. On failure the operator reruns once with flash attention off
    (variant "flash_attn_off", a config change made before the freeze); if that also fails the
    stack is labelled "non_deterministic" and the floor is published as jitter (D12).

    Every execution's transcript is kept under ``out_dir/gate/<variant>/<regime>/r<k>/`` and
    listed in ``executions.jsonl`` beside them; ``out_dir/determinism.json`` holds the result
    of the latest attempt with a summary of every attempt, so a failed default gate and the
    flash_attn_off rerun are both on record. ``provider`` defaults to the model's live
    provider, built and closed inside the gate's event loop.
    """
    import datetime

    from islands_harness.config import canonical_bytes, sha256_bytes

    repo_root = Path(repo_root or Path.cwd())
    out_dir = (
        Path(out_dir) if out_dir is not None else repo_root / config.snapshot.outputs / model.id
    )
    result, executions = asyncio.run(
        _gate_async(
            config,
            model,
            provider,
            repo_root=repo_root,
            out_dir=out_dir,
            variant=variant,
            progress=progress,
        )
    )
    result = GateResult(
        **{
            **asdict(result),
            "regimes": tuple(result.regimes),
            "server": server or {},
            "gpu": gpu,
            "config_hash": sha256_bytes(canonical_bytes(config)),
            "run_on": datetime.date.today().isoformat(),
        }
    )
    base = out_dir / "gate" / variant
    base.mkdir(parents=True, exist_ok=True)
    (base / "executions.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in executions),
        encoding="utf-8",
        newline="\n",
    )
    path = out_dir / "determinism.json"
    attempts = []
    if path.is_file():
        attempts = [
            a
            for a in json.loads(path.read_text(encoding="utf-8")).get("attempts", [])
            if a.get("variant") != variant
        ]
    attempts.append(
        {
            "variant": variant,
            "label": result.label,
            "passed": result.passed,
            "disagreeing_specs": len(result.disagreeing_specs),
            "run_on": result.run_on,
        }
    )
    payload = {**asdict(result), "attempts": attempts}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return result


# --------------------------------------------------------------------------------------
# Noise floor
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JitterResult:
    protocol: str  # "fixed_seed" | "varied_seed" | "hosted_block"
    cell: dict[str, float]
    reruns: int
    n_per_rerun: int
    regime: str
    identity_rate: float | None  # share of reruns whose hashes equal rerun 1 (fixed_seed)
    flip_rate: float | None  # share of documents whose outcome is not constant across reruns
    flip_lo: float | None
    flip_hi: float | None
    jitter_sd: float  # SD of p_r over reruns
    binomial_sd: float | None  # sqrt(p (1 - p) / n) at the mean rate (varied_seed, hosted)
    phi: float | None  # jitter_sd / binomial_sd
    rates: list[float] = field(default_factory=list)


def noise_floor(
    config: SnapshotConfig, model: ModelConfig, provider: ChatProvider
) -> list[JitterResult]:
    """The fixed-seed and varied-seed protocols locally, the three-cell block hosted.

    Fixed seed: the local floor cell's specs executed ``fixed_seed_reruns`` times at the
    sweep's concurrency in round-robin order, never serially. Per rerun r: p_r, transcript
    hashes. Report identity rate, flip rate with its Wilson interval, jitter SD = SD(p_r).
    Varied seed: ``varied_seed_reruns`` reruns with seed_offset = r; report SD(p_r) beside
    sqrt(p (1 - p) / n) and phi. Hosted: each pre-registered cell, ``reruns`` times, on the
    first ``documents_per_rerun`` indices (the block reading). Writes jitter.json.

    TODO(M7): implement on top of specs.noise_floor_specs and runner.run_specs.
    """
    raise NotImplementedError("TODO(M7): noise floor")


# --------------------------------------------------------------------------------------
# Probe, estimate, spend
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    model_id: str
    points: list[dict[str, Any]]  # tools, fault, runs, mean input/output/cache tokens, mean turns
    mean_input_tokens: float
    mean_output_tokens: float
    mean_cache_read_tokens: float
    mean_turns: float
    measured_on: str
    mean_cache_write_tokens: float = 0.0  # Anthropic bills cache writes at 1.25x input


@dataclass(frozen=True)
class SpendPlan:
    model_id: str
    price_table: str
    runs: dict[str, int]  # phase -> run count
    usd: dict[str, float]  # phase -> estimated USD
    total_usd: float
    cap_usd: float | None
    within_cap: bool


def forecast_from_runs(rows: list[dict[str, Any]], model_id: str, measured_on: str) -> Forecast:
    """Mean tokens and turns per run, per probe point and overall, from runs.jsonl rows.
    Runs that ended aborted_transport are excluded: they measure the network, not the task."""
    usable = [r for r in rows if r["outcome"] != "aborted_transport"]
    if not usable:
        raise ValueError("no usable probe runs to forecast from")

    def means(group: list[dict[str, Any]]) -> dict[str, float]:
        n = len(group)
        return {
            "mean_input_tokens": sum(r["usage"]["input_tokens"] for r in group) / n,
            "mean_output_tokens": sum(r["usage"]["output_tokens"] for r in group) / n,
            "mean_cache_read_tokens": sum(r["usage"]["cache_read_tokens"] for r in group) / n,
            "mean_cache_write_tokens": sum(r["usage"]["cache_write_tokens"] for r in group) / n,
            "mean_turns": sum(r["turns"] for r in group) / n,
        }

    points = []
    for key in sorted({(r["tools"], r["fault_rate"]) for r in usable}):
        group = [r for r in usable if (r["tools"], r["fault_rate"]) == key]
        points.append({"tools": key[0], "fault": key[1], "runs": len(group), **means(group)})
    return Forecast(model_id=model_id, points=points, measured_on=measured_on, **means(usable))


async def _probe_async(
    config: SnapshotConfig,
    model: ModelConfig,
    provider: ChatProvider,
    repo_root: Path,
    out_dir: Path,
    measured_on: str,
    progress: Any = None,
) -> Forecast:
    from islands_harness.runner import SpendLedger, Storage, load_prices, load_task, run_specs
    from islands_harness.specs import probe_specs

    task = load_task(config, repo_root)
    storage = Storage(out_dir)
    ledger = None
    if model.prices is not None:
        ledger = SpendLedger(
            prices=load_prices(repo_root / model.prices, model.model), cap_usd=model.spend_cap_usd
        )
    specs = probe_specs(config, model, sorted(task.documents), config.sweep.mixes_to_run[0])
    tau = load_prereg(repo_root / config.snapshot.preregistration).success.tau
    await run_specs(
        specs,
        provider,
        task,
        storage,
        ledger,
        config=config,
        model=model,
        tau=tau,
        progress=progress,
    )
    ids = {s.run_id for s in specs}
    rows = [
        json.loads(line)
        for line in storage.runs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    forecast = forecast_from_runs([r for r in rows if r["run_id"] in ids], model.id, measured_on)
    (Path(out_dir) / "forecast.json").write_text(
        json.dumps(asdict(forecast), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return forecast


def probe(
    config: SnapshotConfig,
    model: ModelConfig,
    provider: ChatProvider,
    *,
    repo_root: Path,
    out_dir: Path,
    measured_on: str,
    progress: Any = None,
) -> Forecast:
    """``probe.runs_per_point`` runs at each probe point; measures tokens and turns per run.
    Writes forecast.json (for the deposit) beside the probe's runs.jsonl in ``out_dir``.
    Resumable like a sweep: probe run ids are distinct from sweep run ids."""
    return asyncio.run(
        _probe_async(config, model, provider, repo_root, out_dir, measured_on, progress)
    )


def is_hosted(model: ModelConfig) -> bool:
    """A model served from outside this machine: Anthropic, or an OpenAI-compatible base URL
    that is not loopback. Hosted models get the hosted noise floor and cost money."""
    if model.provider == "anthropic":
        return True
    host = (model.base_url or "").split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    return host not in ("127.0.0.1", "localhost", "::1", "[::1]")


def phase_counts(config: SnapshotConfig, model: ModelConfig) -> dict[str, int]:
    """Planned runs per phase for one model and the configured mixes, from the config alone."""
    fault_rungs = len(config.sweep.fault_axis)
    in_scope = len(config.sweep.in_scope_tools) * fault_rungs * model.documents * model.epochs
    rung1 = (1 in config.sweep.tools_axis) * fault_rungs * config.sweep.rung1_runs_per_cell
    counts = {
        "calibration": config.calibration.pilot_runs * len(config.calibration.difficulty_levels),
        "probe": len(config.probe.tools_points)
        * len(config.probe.fault_points)
        * config.probe.runs_per_point,
        "sweep": (in_scope + rung1) * len(config.sweep.mixes_to_run),
    }
    if is_hosted(model):
        h = config.noise_floor.hosted
        counts["noise_floor"] = len(h.cells) * h.reruns * h.documents_per_rerun
    else:
        loc = config.noise_floor.local
        counts["noise_floor"] = (
            loc.fixed_seed_reruns * (loc.fixed_seed_documents or model.documents)
            + loc.varied_seed_reruns * model.documents
        )
        g = config.determinism_gate
        counts["determinism_gate"] = g.specs * g.replays_per_regime * 3
    return counts


def estimate(
    forecast: Forecast,
    prices: dict[str, float],
    counts: dict[str, int],
    cap_usd: float | None,
    *,
    price_table: str = "",
) -> SpendPlan:
    """Measured tokens per run times the price table times run counts per phase. A phase's
    cost is runs x (input x p_in + output x p_out + cache_read x p_cache_read + cache_write x
    p_cache_write), with every token count taken as the probe's mean per run. Within cap
    means at or under 95 percent of the cap, the same line the runner stops at."""
    per_run = (
        forecast.mean_input_tokens * prices["input"]
        + forecast.mean_output_tokens * prices["output"]
        + forecast.mean_cache_read_tokens * prices["cache_read"]
        + forecast.mean_cache_write_tokens * prices["cache_write"]
    )
    usd = {phase: round(n * per_run, 2) for phase, n in counts.items()}
    total = round(sum(usd.values()), 2)
    return SpendPlan(
        model_id=forecast.model_id,
        price_table=price_table,
        runs=dict(counts),
        usd=usd,
        total_usd=total,
        cap_usd=cap_usd,
        within_cap=cap_usd is None or total <= 0.95 * cap_usd,
    )


# --------------------------------------------------------------------------------------
# Doctor
# --------------------------------------------------------------------------------------


@dataclass
class DoctorReport:
    checks: list[dict[str, Any]] = field(default_factory=list)  # name, ok, blocking, detail
    outbound_hosts: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return any(c["blocking"] and not c["ok"] for c in self.checks)

    def render(self) -> str:
        lines = [
            f"[{'ok' if c['ok'] else 'FAIL' if c['blocking'] else 'warn'}] {c['name']}: {c['detail']}"
            for c in self.checks
        ]
        lines.append(
            "outbound hosts the config would contact: " + (", ".join(self.outbound_hosts) or "none")
        )
        return "\n".join(lines)


def _check(name: str, ok: bool, detail: str, *, blocking: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": ok, "blocking": blocking, "detail": detail}


def _check_python() -> dict[str, Any]:
    import sys

    v = sys.version_info
    return _check(
        "python", (v.major, v.minor) == (3, 12), f"{v.major}.{v.minor}.{v.micro} (3.12 required)"
    )


def _check_uv(repo_root: Path) -> dict[str, Any]:
    import os
    import re
    import shutil
    import subprocess

    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        return _check("uv", False, "uv not found; install it with winget (README step 1)")
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    pin = re.search(r'required-version\s*=\s*"==([^"]+)"', pyproject)
    try:
        version = subprocess.run(
            [uv, "--version"], capture_output=True, text=True, timeout=30, check=True
        ).stdout.split()[1]
        locked = subprocess.run(
            [uv, "lock", "--check", "--offline"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return _check("uv", False, f"uv did not run: {exc}")
    problems = []
    if pin and version != pin.group(1):
        problems.append(f"uv {version}, pyproject requires {pin.group(1)}")
    if locked.returncode != 0:
        problems.append("uv.lock is out of date with pyproject.toml")
    return _check(
        "uv and lock",
        not problems,
        "; ".join(problems) or f"uv {version}; uv.lock matches pyproject.toml",
    )


def _check_dataset(config: SnapshotConfig, repo_root: Path) -> dict[str, Any]:
    from islands_harness import dataset as ds
    from islands_harness.config import LockMismatch

    try:
        lock = ds.verify_lock(repo_root / config.task.dir / config.task.dataset)
    except LockMismatch as exc:
        return _check("dataset", False, str(exc))
    return _check("dataset", True, f"{len(lock['files'])} files match LOCK.json")


def _check_llama_build(config: SnapshotConfig) -> dict[str, Any] | None:
    import os

    builds = {m.expect.build: m.expect.build_commit for m in config.models if m.expect is not None}
    if not builds:
        return None
    problems, found = [], []
    for build, commit in sorted(builds.items()):
        path = (
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "islands"
            / "llama.cpp"
            / build
            / "build.json"
        )
        if not path.is_file():
            problems.append(f"llama.cpp {build} is not installed (server/llamacpp/install.ps1)")
            continue
        info = json.loads(path.read_text(encoding="utf-8-sig"))
        if info.get("commit") != commit or not Path(str(info.get("server_exe", ""))).is_file():
            problems.append(
                f"llama.cpp {build}: commit or llama-server.exe does not match the config"
            )
        else:
            found.append(f"{build} ({str(commit)[:9]}, CUDA {info.get('cuda')})")
    return _check(
        "llama.cpp build", not problems, "; ".join(problems) or "installed: " + ", ".join(found)
    )


def _check_gpu(gpu: dict[str, Any] | None, config: SnapshotConfig) -> dict[str, Any] | None:
    if not any(m.expect is not None for m in config.models):
        return None
    if gpu is None:
        return _check(
            "gpu", False, "nvidia-smi did not answer; the local models need an NVIDIA GPU"
        )
    return _check(
        "gpu",
        True,
        f"{gpu['name']}, driver {gpu['driver']}, {gpu['used_mib']} of {gpu['total_mib']} MiB in use",
    )


def _check_local_model(
    model: ModelConfig, repo_root: Path, gpu: dict[str, Any] | None, rehash: bool
) -> list[dict[str, Any]]:
    from islands_harness import localstack

    checks = []
    ok, detail = localstack.verify_weights(repo_root, model, rehash=rehash)
    checks.append(_check(f"{model.id} weights", ok, detail))
    running, diffs, _ = localstack.server_differences(model)
    size_mib = localstack.weights_path(repo_root, model).stat().st_size / 2**20 if ok else 0.0
    if running:
        checks.append(
            _check(f"{model.id} vram", True, "the server has loaded the model on this GPU")
        )
    elif gpu is not None:
        need = size_mib + 2560  # f16 KV cache at 16k context and compute buffers, generously
        checks.append(
            _check(
                f"{model.id} vram",
                need <= gpu["total_mib"],
                f"needs about {need:,.0f} MiB of {gpu['total_mib']:,} MiB",
            )
        )
    if not running:
        checks.append(
            _check(
                f"{model.id} server",
                False,
                f"not running; start it with `islands serve start {model.model}` (one model at a time)",
                blocking=False,
            )
        )
    else:
        checks.append(
            _check(
                f"{model.id} server",
                not diffs,
                "; ".join(diffs)
                or "build, slots, context, weights, quantization, chat template and every flag match the config",
            )
        )
    return checks


def doctor(
    config: SnapshotConfig,
    repo_root: Path | None = None,
    *,
    model_id: str | None = None,
    rehash: bool = False,
) -> DoctorReport:
    """Check the toolchain (uv, Python 3.12, lock in sync), free VRAM against each local
    weights file, each weights sha256 against models/*.lock.json, API keys named by the config,
    each endpoint's reachability, and for llama.cpp the running server's ``GET /props``
    against the model's ``expect`` block (build_info, model_path, total_slots == 1, n_ctx,
    chat format); then list every outbound host from
    ``allowed_hosts`` (the whole perimeter).

    Each check is one function returning {name, ok, blocking, detail}. A server that is not
    running is a warning, not a failure, because only one local model runs at a time; a
    server that runs with anything different from the config is a failure. No check contacts
    a hosted endpoint: for hosted models doctor only checks that the key is set, so the first
    request to a paid API is always an explicitly approved run.
    """
    import os

    from islands_harness import localstack

    repo_root = Path(repo_root or Path.cwd())
    report = DoctorReport()
    report.checks.extend([_check_python(), _check_uv(repo_root), _check_dataset(config, repo_root)])
    for item in (_check_llama_build(config), _check_gpu(gpu := localstack.gpu_memory(), config)):
        if item is not None:
            report.checks.append(item)
    for model in config.models:
        if model_id is not None and model.id != model_id:
            continue
        if model.expect is not None:
            report.checks.extend(_check_local_model(model, repo_root, gpu, rehash))
        elif model.api_key_env:
            present = bool(os.environ.get(model.api_key_env))
            report.checks.append(
                _check(
                    f"{model.id} key",
                    present,
                    f"{model.api_key_env} is set (not contacted)"
                    if present
                    else f"{model.api_key_env} is not set; {model.id} cannot run until it is",
                    blocking=False,
                )
            )
    report.outbound_hosts = sorted({h for m in config.models for h in m.allowed_hosts})
    return report


# --------------------------------------------------------------------------------------
# Selftest and replay
# --------------------------------------------------------------------------------------


def load_task_module(path: Path) -> Any:
    """Import a task file (the grader) by path; the public task lives outside the package.
    One loader for every task file (tools.load_module_from_path)."""
    from islands_harness.tools import load_module_from_path

    return load_module_from_path(Path(path))


def read_prompt(path: Path) -> str:
    """The system prompt as sent: UTF-8, CRLF normalized to LF, nothing added or stripped."""
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def fresh_output_dir(out_dir: Path, marker: str) -> Path:
    """Prepare a tool-owned output directory. A directory this tool created before (it holds
    ``marker``) has its runs and transcripts cleared; any other non-empty directory is refused,
    so a mistyped path can never clear someone else's results."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not (out / marker).exists():
        raise FileExistsError(
            f"{out} is not empty and was not created by this command; choose another path"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.jsonl").unlink(missing_ok=True)
    if (out / "transcripts").exists():
        shutil.rmtree(out / "transcripts")
    (out / marker).write_text("created by the islands harness; safe to clear\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------
# Selftest: every loop path on the fake provider, then a whole synthetic snapshot
# --------------------------------------------------------------------------------------

SELFTEST_DOC_ID = "st-0001"
SELFTEST_DOCUMENT = (
    "INVOICE\nInvoice number: ST-0001\nDate: 2026-03-14\nFrom: Selftest Supplies Ltd\n"
    "Item: Widget  Qty 2  Unit price 10.00  Amount 20.00\nTotal due: 20.00 EUR\n"
)
SELFTEST_GOLD = {
    "invoice_number": "ST-0001",
    "invoice_date": "2026-03-14",
    "vendor_name": "Selftest Supplies Ltd",
    "currency": "EUR",
    "total_amount": 20.0,
    "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0}],
}


def _selftest_scenarios() -> list[tuple[str, int, float, list[Any], dict[str, Any], str | None]]:
    """(name, tools, fault rate, script, agent overrides, expected outcome or None if the
    outcome depends on which faults fire)."""
    from islands_harness.providers.base import ScriptedTurn, StopReason, ToolCall

    def call(name: str, args: dict[str, Any], i: int) -> ToolCall:
        return ToolCall(
            id=f"st{i}", name=name, arguments=args, arguments_text=json.dumps(args, sort_keys=True)
        )

    fetch = ScriptedTurn(tool_calls=(call("fetch_document", {"doc_id": SELFTEST_DOC_ID}, 0),))
    submit = ScriptedTurn(tool_calls=(call("submit_record", {"record": SELFTEST_GOLD}, 1),))
    text_record = "Here is the record: " + json.dumps(SELFTEST_GOLD, sort_keys=True)
    return [
        ("submitted", 2, 0.0, [fetch, submit], {}, "submitted"),
        (
            "no_submission",
            2,
            0.0,
            [ScriptedTurn(text="I cannot find an invoice.")],
            {},
            "no_submission",
        ),
        ("limit_turns", 2, 0.0, [fetch] * 3, {"max_turns": 3}, "limit_turns"),
        (
            "limit_tool_calls",
            2,
            0.0,
            [
                ScriptedTurn(
                    tool_calls=tuple(
                        call("fetch_document", {"doc_id": SELFTEST_DOC_ID}, i) for i in range(4)
                    )
                )
            ],
            {"max_tool_calls": 3},
            "limit_tool_calls",
        ),
        ("limit_tokens", 2, 0.0, [fetch], {"token_limit": 50}, "limit_tokens"),
        ("refusal", 2, 0.0, [ScriptedTurn(stop_reason=StopReason.refusal)], {}, "refusal"),
        (
            "limit_output",
            2,
            0.0,
            [ScriptedTurn(tool_calls=fetch.tool_calls, stop_reason=StopReason.max_tokens)],
            {},
            "limit_output",
        ),
        ("context_overflow", 2, 0.0, [ScriptedTurn(raise_overflow=True)], {}, "context_overflow"),
        (
            "aborted_transport",
            2,
            0.0,
            [ScriptedTurn(raise_transport=True)],
            {},
            "aborted_transport",
        ),
        (
            "faults_at_50_percent",
            6,
            0.5,
            [fetch, submit, submit, submit, submit, submit],
            {"max_turns": 6},
            None,
        ),
        ("rung1_text_record", 1, 0.0, [ScriptedTurn(text=text_record)], {}, "no_submission"),
    ]


async def _selftest_async(config: SnapshotConfig, out_dir: Path, repo_root: Path) -> dict[str, Any]:
    from islands_harness.faults import FaultInjector
    from islands_harness.loop import Prompts, run_one
    from islands_harness.providers.base import FakeProvider, Sampling
    from islands_harness.runner import Storage, grade_run
    from islands_harness.specs import Cell, RunSpec, run_id
    from islands_harness.tools import Registry, RunContext

    task_dir = repo_root / config.task.dir
    registry = Registry.load(
        task_dir / "tools.py",
        required=config.task.required_tools,
        mixes=config.task.mixes,
        rung1=config.task.rung1,
    )
    grader = load_task_module(task_dir / config.task.grader)
    prompts = Prompts(system=read_prompt(task_dir / config.task.system_prompt))
    gold = GoldRecordLike(SELFTEST_DOC_ID, SELFTEST_GOLD)
    storage = Storage(fresh_output_dir(out_dir, ".selftest"))
    sampling = Sampling(
        temperature=0.7, top_p=1.0, max_tokens=256, seed=None, thinking=None, effort=None
    )

    async def one(
        i: int, name: str, tools: int, rate: float, script: list[Any], over: dict[str, Any]
    ) -> tuple[Any, list[Message]]:
        cell = Cell(tools, rate, "A")
        spec = RunSpec(
            run_id=run_id("selftest-fake", cell, i, "selftest", None),
            model_id="selftest-fake",
            cell=cell,
            index=i,
            doc_id=SELFTEST_DOC_ID,
            epoch=0,
            sample_seed=i,
            tool_list=tuple(t.name for t in registry.for_rung("A", tools)),
            phase="selftest",
        )
        ctx = RunContext(
            doc_id=SELFTEST_DOC_ID, document_text=SELFTEST_DOCUMENT, dataset_dir=task_dir, tools={}
        )
        agent = config.agent.model_copy(update=over)
        injector = FaultInjector(config.faults, spec, config.snapshot.root_seed)
        return await run_one(
            spec,
            FakeProvider(list(script), model_id="selftest-fake"),
            registry,
            injector,
            agent,
            prompts,
            ctx=ctx,
            sampling=sampling,
        )

    outcomes: dict[str, str] = {}
    for i, (name, tools, rate, script, over, expected) in enumerate(_selftest_scenarios()):
        record, transcript = await one(i, name, tools, rate, script, over)
        record.transcript_path = (
            storage.write_transcript(record.run_id, transcript).relative_to(storage.dir).as_posix()
        )
        grade_run(record, grader, gold, tau=0.8)
        storage.append_run(record)
        outcomes[name] = record.outcome.value
        if expected is not None and record.outcome.value != expected:
            raise AssertionError(
                f"selftest {name}: expected {expected}, got {record.outcome.value}"
            )
        if name == "submitted" and record.score != 1.0:
            raise AssertionError(
                f"selftest submitted: the gold record scored {record.score}, not 1.0"
            )
        if name == "rung1_text_record" and record.submission_attempted_in_text is not True:
            raise AssertionError("selftest rung1_text_record: the text submission was not detected")

    rows = [json.loads(line) for line in storage.runs_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(outcomes) or len({r["run_id"] for r in rows}) != len(rows):
        raise AssertionError("selftest: runs.jsonl does not hold exactly one line per scenario")
    for row in rows:
        json.loads((storage.dir / row["transcript_path"]).read_text(encoding="utf-8"))
    name, tools, rate, script, over, _ = _selftest_scenarios()[0]
    _, first = await one(0, name, tools, rate, script, over)
    _, second = await one(0, name, tools, rate, script, over)
    if canonical_transcript_hash(first) != canonical_transcript_hash(second):
        raise AssertionError("selftest: the same scripted run hashed differently twice")
    snapshot = await _selftest_snapshot(config, storage.dir, repo_root)
    return {
        "out_dir": str(storage.dir),
        "runs": len(rows),
        "outcomes": outcomes,
        "snapshot": snapshot,
    }


SELFTEST_MODEL_ID = "selftest-synthetic"
SELFTEST_DOCUMENTS = 40
SELFTEST_SNAPSHOT_ID = "selftest"
SELFTEST_EXPECTED = {"instrument": "in_calibration", "p1": "survives", "p2": "survives"}


def selftest_model(config: SnapshotConfig) -> ModelConfig:
    """The synthetic model's config: a local model's settings under the selftest id, 40
    documents, one epoch, no server expectations (no server is contacted)."""
    from islands_harness.report import SYNTHETIC_MODEL

    base = next(m for m in config.models if m.provider == "openai_compat")
    return base.model_copy(
        update={
            "id": SELFTEST_MODEL_ID,
            "model": SYNTHETIC_MODEL,
            "documents": SELFTEST_DOCUMENTS,
            "epochs": 1,
            "concurrency": 1,
            "expect": None,
            "lock": None,
        }
    )


async def _selftest_snapshot(
    config: SnapshotConfig, out_dir: Path, repo_root: Path
) -> dict[str, Any]:
    """A whole snapshot from the synthetic agent: the full 6 by 6 sweep on 40 real task
    documents through the runner, analysis, report and bundle, then the reviewer's checks
    on the result. Raises AssertionError on any deviation."""
    from islands_harness.config import prereg_hash
    from islands_harness.providers.synthetic import SyntheticAgent
    from islands_harness.report import RunRowV1, bundle, check_sums, report
    from islands_harness.runner import Storage, load_task, run_specs
    from islands_harness.specs import expand_cells
    from islands_harness.specs import run_specs as specs_for_cell
    from islands_harness.stats.analyze import analyze, read_runs
    from islands_harness.verification import check_analysis, regrade

    snap_dir = Path(out_dir) / "snapshot"
    if snap_dir.exists():
        shutil.rmtree(snap_dir)  # inside the selftest's own, marker-checked output directory
    model = selftest_model(config)
    model_dir = snap_dir / model.id
    task = load_task(config, repo_root)
    prereg = load_prereg(repo_root / config.snapshot.preregistration)
    agent = SyntheticAgent(
        {d: g.fields for d, g in task.gold.items()}, task.documents, model_id=model.id
    )
    specs = [
        s
        for cell in expand_cells(config, "A")
        for s in specs_for_cell(config, model, cell, sorted(task.documents))
    ]
    await run_specs(
        specs,
        agent,
        task,
        Storage(model_dir),
        None,
        config=config,
        model=model,
        tau=prereg.success.tau,
        concurrency=1,
        resume=False,
    )
    rows, torn = read_runs(model_dir / "runs.jsonl")
    if len(rows) != len(specs) or torn:
        raise AssertionError(f"selftest snapshot: {len(rows)} rows for {len(specs)} specs")
    for row in rows:
        RunRowV1.model_validate(row)
    results = analyze(
        rows,
        prereg,
        snapshot_id=SELFTEST_SNAPSHOT_ID,
        model_id=model.id,
        pairing="exact_synthetic_agent",
        prereg_hash=prereg_hash(prereg),
        out_dir=model_dir,
        torn_lines=torn,
    )
    got = {
        "instrument": results["instrument"]["label"],
        "p1": results["verdicts"]["p1"]["verdict"],
        "p2": results["verdicts"]["p2"]["verdict"],
    }
    if got != SELFTEST_EXPECTED:
        raise AssertionError(
            f"selftest snapshot: expected {SELFTEST_EXPECTED}, got {got}: {results['statement']}"
        )
    snapshot = report(
        snap_dir, config=config, repo_root=repo_root, kind="selftest", models={model.id: model}
    )
    check = check_analysis(model_dir, config=config, repo_root=repo_root)
    if not check.prereg_matches or any(f.status != "identical" for f in check.files):
        raise AssertionError(f"selftest snapshot: re-analysis differs: {check}")
    graded = regrade(model_dir, config=config, repo_root=repo_root)
    if not graded.ok or graded.checked != len(rows):
        raise AssertionError(f"selftest snapshot: re-grading failed: {graded}")
    picks = [
        next(r for r in rows if r.get("fault_events") and r.get("recovered")),
        next(r for r in rows if r.get("fault_events") and r.get("recovered") is False),
        next(r for r in rows if int(r["tools"]) == 1),
        next(r for r in rows if not r.get("fault_events") and int(r["tools"]) == 6),
    ]
    for row in picks:
        diff = await replay_async(
            row["run_id"], model_dir, agent, config=config, repo_root=repo_root, model=model
        )
        if not diff.identical:
            raise AssertionError(f"selftest snapshot: replay of {row['run_id']}: {diff.detail}")
    built = bundle(snap_dir)
    sums = check_sums(snap_dir)
    if sums["mismatched"] or sums["missing"] or sums["unlisted"]:
        raise AssertionError(f"selftest snapshot: SHA256SUMS does not verify: {sums}")
    return {
        "dir": str(snap_dir),
        "snapshot_json": str(snap_dir / "snapshot.json"),
        "runs": len(rows),
        "statement": snapshot.models[0].statement,
        "replays": len(picks),
        "regraded": graded.checked,
        "bundle_members": built.members,
    }


@dataclass(frozen=True)
class GoldRecordLike:
    """The selftest's gold record, shaped like dataset.GoldRecord."""

    doc_id: str
    fields: dict[str, Any]


def selftest(
    config: SnapshotConfig, out_dir: Path, repo_root: Path | None = None
) -> dict[str, Any]:
    """The pipeline on the fake provider with the real task tools, prompt and grader.

    First every loop termination path, a faulted run and rung 1, graded and stored, with a
    determinism check of the harness itself (the same scripted run hashes the same twice).
    Then a whole synthetic snapshot under ``out_dir/snapshot``: the full sweep of the
    synthetic agent (providers/synthetic.py) on 40 task documents, analysis, report
    (``snapshot.json`` validated against snapshot.v1) and bundle, followed by the reviewer's
    checks: re-analysis byte-identical, every run re-graded, four runs replayed with
    identical transcripts, and SHA256SUMS verified. The synthetic surface is known in
    advance, so the verdicts must come out as ``SELFTEST_EXPECTED``. Raises AssertionError
    on any deviation."""
    return asyncio.run(_selftest_async(config, Path(out_dir), repo_root or Path.cwd()))


# --------------------------------------------------------------------------------------
# Smoke: four real runs against a live model endpoint
# --------------------------------------------------------------------------------------

SMOKE_CELLS = ((2, 0.0), (2, 0.5), (6, 0.0), (6, 0.5))


async def _smoke_async(
    config: SnapshotConfig, model: ModelConfig, out_dir: Path, repo_root: Path, doc_index: int
) -> list[dict[str, Any]]:
    from islands_harness import dataset as ds
    from islands_harness.faults import FaultInjector
    from islands_harness.loop import Prompts, run_one
    from islands_harness.providers.factory import build_provider, sampling_for
    from islands_harness.runner import Storage, grade_run
    from islands_harness.specs import Cell, run_specs
    from islands_harness.tools import Registry, RunContext

    task_dir = repo_root / config.task.dir
    data_dir = task_dir / config.task.dataset
    docs = {d.id: d.text for d in ds.load(data_dir)}
    gold = ds.load_gold(data_dir)
    registry = Registry.load(
        task_dir / "tools.py",
        required=config.task.required_tools,
        mixes=config.task.mixes,
        rung1=config.task.rung1,
    )
    grader = load_task_module(task_dir / config.task.grader)
    prompts = Prompts(system=read_prompt(task_dir / config.task.system_prompt))
    tau = load_prereg(repo_root / config.snapshot.preregistration).success.tau
    storage = Storage(fresh_output_dir(out_dir, ".smoke"))
    provider = build_provider(model, config.agent)
    rows: list[dict[str, Any]] = []
    try:
        for tools, rate in SMOKE_CELLS:
            spec = run_specs(config, model, Cell(tools, rate, "A"), sorted(docs), phase="smoke")[
                doc_index
            ]
            ctx = RunContext(
                doc_id=spec.doc_id, document_text=docs[spec.doc_id], dataset_dir=data_dir, tools={}
            )
            injector = FaultInjector(config.faults, spec, config.snapshot.root_seed)
            record, transcript = await run_one(
                spec,
                provider,
                registry,
                injector,
                config.agent,
                prompts,
                ctx=ctx,
                sampling=sampling_for(model, spec.sample_seed),
            )
            record.transcript_path = (
                storage.write_transcript(spec.run_id, transcript)
                .relative_to(storage.dir)
                .as_posix()
            )
            grade_run(record, grader, gold[spec.doc_id], tau)
            storage.append_run(record)
            rows.append(
                {
                    "cell": f"{tools} tools, {round(rate * 100)}% faults",
                    "doc_id": spec.doc_id,
                    "outcome": record.outcome.value,
                    "turns": record.turns,
                    "tool_calls": record.tool_calls,
                    "faults": len(record.fault_events),
                    "malformed": record.malformed_calls,
                    "score": record.score,
                    "success": record.success,
                    "seconds": record.seconds,
                    "tokens": record.usage.total,
                    "transcript": record.transcript_path,
                }
            )
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()
    return rows


def smoke(
    config: SnapshotConfig,
    model: ModelConfig,
    out_dir: Path,
    repo_root: Path | None = None,
    *,
    doc_index: int = 0,
) -> list[dict[str, Any]]:
    """One run at each of rungs 2 and 6 at 0 and 50 percent faults on one document, against
    the live endpoint the model config names. Writes runs.jsonl and transcripts to
    ``out_dir`` (exploratory; nothing here can enter a snapshot) and returns one summary row
    per run, including seconds per run, which schedules the sweep."""
    return asyncio.run(
        _smoke_async(config, model, Path(out_dir), repo_root or Path.cwd(), doc_index)
    )


@dataclass(frozen=True)
class ReplayDiff:
    run_id: str
    identical: bool
    original_hash: str
    replay_hash: str
    first_divergence: int | None  # message index
    original_outcome: str
    replay_outcome: str
    detail: str


def rebuild_spec(
    row: dict[str, Any], config: SnapshotConfig, model: ModelConfig, doc_ids: list[str]
) -> Any:
    """The RunSpec a runs.jsonl row was executed from, rebuilt with specs.run_specs from the
    row's cell and index and never from the transcript. Refuses when the rebuilt spec does not
    reproduce the row's run id, document and seed (a changed config or dataset)."""
    from islands_harness.specs import Cell, run_specs

    cell = Cell(int(row["tools"]), float(row["fault_rate"]), str(row["mix"]))
    specs = run_specs(
        config,
        model,
        cell,
        doc_ids,
        phase=str(row["phase"]),
        seed_offset=row.get("seed_offset"),
    )
    index = int(row["index"])
    spec = specs[index] if index < len(specs) else None
    if (
        spec is None
        or spec.run_id != row["run_id"]
        or spec.doc_id != row["doc_id"]
        or spec.sample_seed != row["sample_seed"]
    ):
        raise ValueError(
            f"run {row['run_id']}: the recorded spec cannot be rebuilt from the current config "
            "and dataset (run id, document or seed differs)"
        )
    return spec


async def replay_async(
    run_id: str,
    results_dir: Path,
    provider: ChatProvider,
    *,
    config: SnapshotConfig,
    repo_root: Path,
    model: ModelConfig | None = None,
) -> ReplayDiff:
    """``replay`` inside a running event loop."""
    from islands_harness.runner import execute_spec, load_task, read_transcript
    from islands_harness.stats.analyze import read_runs

    results_dir = Path(results_dir)
    rows = {r["run_id"]: r for r in read_runs(results_dir / "runs.jsonl")[0]}
    if run_id not in rows:
        raise KeyError(f"no run {run_id} in {results_dir / 'runs.jsonl'}")
    row = rows[run_id]
    model = model or config.model_by_id(str(row["model_id"]))
    task = load_task(config, repo_root)
    spec = rebuild_spec(row, config, model, sorted(task.documents))
    record, transcript = await execute_spec(spec, provider, task, config=config, model=model)
    original = read_transcript(results_dir / row["transcript_path"])
    a, b = canonical_rows(original), canonical_rows(transcript)
    first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    ha, hb = canonical_transcript_hash(original), canonical_transcript_hash(transcript)
    detail = (
        "identical transcript"
        if ha == hb
        else f"transcripts diverge at message {first} of {len(a)} (original) and {len(b)} (replay)"
    )
    return ReplayDiff(
        run_id=run_id,
        identical=ha == hb,
        original_hash=ha,
        replay_hash=hb,
        first_divergence=first,
        original_outcome=str(row["outcome"]),
        replay_outcome=record.outcome.value,
        detail=detail,
    )


def replay(
    run_id: str,
    results_dir: Path,
    provider: ChatProvider,
    *,
    config: SnapshotConfig,
    repo_root: Path,
    model: ModelConfig | None = None,
) -> ReplayDiff:
    """Re-execute one run from its recorded spec (the runs.jsonl row plus the frozen inputs)
    and diff the canonical transcript against the stored one. On a stack the gate verified
    the two are identical; the first divergent message is reported otherwise."""
    return asyncio.run(
        replay_async(run_id, results_dir, provider, config=config, repo_root=repo_root, model=model)
    )


# --------------------------------------------------------------------------------------
# Calibration: the anchor cell per difficulty level, on pilot documents
# --------------------------------------------------------------------------------------


def pilot_seed(config: SnapshotConfig, difficulty: int) -> int:
    """The seed of one difficulty level's pilot dataset. Derived from the root seed, so it is
    reproducible, and never the snapshot dataset's seed, so no pilot document is a snapshot
    document: calibration cannot tune the threshold on the documents it will be measured on."""
    from islands_harness.rng import seed64

    return seed64("pilot", config.snapshot.root_seed, difficulty) % (2**31)


def rate_table(
    rows: list[dict[str, Any]], taus: list[float], band: tuple[float, float]
) -> dict[str, dict[str, Any]]:
    """Per candidate tau: successes (score >= tau, tau rounded to six decimals as in grading),
    graded runs, the rate, its Wilson interval, and whether the rate lies in the band."""
    from islands_harness.stats.fit import wilson

    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    n = len(scores)
    out: dict[str, dict[str, Any]] = {}
    for tau in taus:
        k = sum(1 for s in scores if s >= round(tau, 6))
        lo, hi = wilson(k, n)
        p = k / n if n else float("nan")
        out[f"{tau:g}"] = {
            "k": k,
            "n": n,
            "p": p,
            "lo": lo,
            "hi": hi,
            "in_band": bool(n) and band[0] <= p <= band[1],
        }
    return out


async def _calibrate_async(
    config: SnapshotConfig,
    model: ModelConfig,
    provider: ChatProvider | None,
    *,
    repo_root: Path,
    out_dir: Path,
    levels: list[int],
    progress: Any = None,
) -> dict[str, Any]:
    from collections import Counter
    from statistics import mean

    from islands_harness import dataset as ds
    from islands_harness.providers.factory import build_provider
    from islands_harness.runner import Storage, TaskBundle, load_task, run_specs
    from islands_harness.specs import Cell
    from islands_harness.specs import run_specs as specs_for_cell
    from islands_harness.stats.analyze import read_runs

    cal = config.calibration
    base = load_task(config, repo_root)
    anchor = Cell(cal.anchor_cell.tools, cal.anchor_cell.fault, config.sweep.mixes_to_run[0])
    own = provider is None
    provider = provider or build_provider(model, config.agent)
    result: dict[str, Any] = {
        "model": model.id,
        "anchor": {"tools": anchor.tools, "fault": anchor.fault_rate, "mix": anchor.mix},
        "pilot_runs": cal.pilot_runs,
        "band": list(cal.target_band),
        "taus": list(cal.tau_candidates),
        "max_malformed_call_rate": cal.max_malformed_call_rate,
        "levels": {},
    }
    try:
        for level in levels:
            pilot = Path(out_dir) / f"pilot-d{level}"
            data_dir = pilot / "dataset"
            if (data_dir / ds.DATASET_LOCK_NAME).exists():
                ds.verify_lock(data_dir)
            else:
                ds.generate(pilot_seed(config, level), cal.pilot_runs, level, data_dir)
            task = TaskBundle(
                registry=base.registry,
                grader=base.grader,
                prompts=base.prompts,
                documents={d.id: d.text for d in ds.load(data_dir)},
                gold=ds.load_gold(data_dir),
                dataset_dir=data_dir,
            )
            specs = specs_for_cell(
                config,
                model,
                anchor,
                sorted(task.documents),
                phase="calibration",
                n_runs=cal.pilot_runs,
            )
            await run_specs(
                specs,
                provider,
                task,
                Storage(pilot),
                None,
                config=config,
                model=model,
                tau=cal.tau_candidates[0],
                resume=True,
            )
            rows, _ = read_runs(pilot / "runs.jsonl")
            calls = sum(int(r.get("tool_calls") or 0) for r in rows)
            malformed = sum(int(r.get("malformed_calls") or 0) for r in rows)
            rate = malformed / calls if calls else 0.0
            scores = [float(r["score"]) for r in rows if r.get("score") is not None]
            result["levels"][str(level)] = {
                "pilot_seed": pilot_seed(config, level),
                "documents": len(task.documents),
                "runs": len(rows),
                "outcomes": dict(sorted(Counter(str(r["outcome"]) for r in rows).items())),
                "mean_score": mean(scores) if scores else None,
                "malformed_call_rate": rate,
                "eligible": rate <= cal.max_malformed_call_rate,
                "taus": rate_table(rows, cal.tau_candidates, cal.target_band),
            }
            if progress is not None:
                at = result["levels"][str(level)]["taus"][f"{cal.tau_candidates[0]:g}"]
                progress(
                    f"difficulty {level}: {at['k']}/{at['n']} at tau {cal.tau_candidates[0]:g}"
                )
    finally:
        if own:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()
    return result


def calibrate(
    config: SnapshotConfig,
    model: ModelConfig,
    provider: ChatProvider | None = None,
    *,
    repo_root: Path | None = None,
    out_dir: Path,
    levels: list[int] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """The pilot at the anchor cell for one model (ARCHITECTURE.md Section 10).

    For each difficulty level, ``pilot_runs`` anchor-cell runs on a pilot dataset of as many
    documents, generated at that level from ``pilot_seed`` under ``out_dir/pilot-d<level>/``
    with its runs and transcripts. Pilot runs never enter a snapshot. Reports, per level and
    candidate tau, the rate with its Wilson interval and whether it lies in the target band,
    and the malformed tool-call rate. Writes ``out_dir/calibration.json``; resumable.
    """
    from islands_harness.stats.verdict import jsonable

    levels = list(levels or config.calibration.difficulty_levels)
    result = asyncio.run(
        _calibrate_async(
            config,
            model,
            provider,
            repo_root=Path(repo_root or Path.cwd()),
            out_dir=Path(out_dir),
            levels=levels,
            progress=progress,
        )
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "calibration.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def recommend_calibration(results: list[dict[str, Any]], config: SnapshotConfig) -> dict[str, Any]:
    """One difficulty level and one tau for the whole snapshot, from every model's pilot.

    Every (tau, difficulty) pair that every model was piloted at is an option. A model is a
    replicate at an option when its anchor rate lies in the target band and its malformed
    tool-call rate is within the limit. Options are ranked by, in order: the number of
    replicates (more first); the tau candidate's position (0.8 first, so the difficulty
    changes before tau does, Section 10); the summed distance of the rates from the band's
    midpoint; the lower difficulty. The first option is the recommendation; models that are
    not replicates there are excluded from snapshot 1, and the method note says why.
    """
    cal = config.calibration
    band = cal.target_band
    mid = (band[0] + band[1]) / 2
    options: list[dict[str, Any]] = []
    for rank, tau in enumerate(cal.tau_candidates):
        key = f"{tau:g}"
        for level in cal.difficulty_levels:
            per_model: dict[str, Any] = {}
            for r in results:
                lvl = (r.get("levels") or {}).get(str(level))
                if lvl is None:
                    break
                cell = lvl["taus"][key]
                # Judged against the config's current band and malformed-call limit, never the
                # flags stored at pilot time: the rule may change after the pilot (r4 did).
                p_rate = cell["p"]
                in_band = p_rate is not None and band[0] <= float(p_rate) <= band[1]
                malformed = lvl.get("malformed_call_rate")
                eligible = (
                    bool(lvl.get("eligible"))
                    if malformed is None
                    else float(malformed) <= cal.max_malformed_call_rate
                )
                per_model[r["model"]] = {
                    "p": p_rate,
                    "lo": cell["lo"],
                    "hi": cell["hi"],
                    "eligible": eligible,
                    "replicate": bool(in_band and eligible),
                }
            else:
                replicates = sorted(m for m, e in per_model.items() if e["replicate"])
                distance = sum(abs(float(e["p"]) - mid) for e in per_model.values())
                options.append(
                    {
                        "difficulty": level,
                        "tau": tau,
                        "replicates": replicates,
                        "excluded": sorted(set(per_model) - set(replicates)),
                        "distance_from_midpoint": distance,
                        "models": per_model,
                        "_rank": (-len(replicates), rank, distance, level),
                    }
                )
    options.sort(key=lambda o: o["_rank"])
    for o in options:
        o.pop("_rank")
    return {"chosen": options[0] if options else None, "options": options}
