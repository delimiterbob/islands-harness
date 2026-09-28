"""Scheduling, storage, grading, spend metering and resume.

The runner takes a list of run specs and drives them through ``loop.run_one`` under a
semaphore, in round-robin cell order (one spec from each cell in turn, so a killed sweep has
partial coverage of every cell rather than complete coverage of a few). Each finished run
is graded, appended as one fsynced line to ``runs.jsonl``, and its transcript written under
``transcripts/<shard>/<run_id>.json``. Spend is metered per run against the price table and
the runner stops at 95 percent of the cap. It never prints a verdict.

Resume semantics: a run's identity is its ``run_id`` (specs.run_id). On start the runner
reads the existing ``runs.jsonl`` and skips every spec whose id is present; a run is in the
file only if its line was fully written and fsynced, so a kill mid-run leaves no partial line
(the write is a single ``os.write`` of the whole line followed by fsync) and the run is
re-executed on resume. Kill-and-resume therefore leaves no duplicate and no gap
(tests/test_harness.py). Runs that ended ``aborted_transport`` are re-executed once on the
same pass; if they abort again they stay in the file with that outcome, are counted in the
manifest and excluded from every denominator. Such a run is written once the next run
completes. When the next run also aborts twice, the stack itself is down (a crashed server,
or one left hung by a hibernation), so the pass stops with ``TransportHalt``, records
neither run, and resume re-executes both once the server is back; without this a dead
server would turn every remaining spec into an excluded row (added 2026-09-27, after a
hibernation stalled the noise floor). Only transport aborts count toward the halt: a run
the server rejected as unparseable model output (``abort_cause``) proves the server is
alive, so it is recorded at once and never halts a pass (added 2026-09-28, after
gpt-oss-20b's deterministic parse failures halted the sweep on the same two runs).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from islands_harness.config import AgentConfig, ModelConfig, SnapshotConfig
from islands_harness.dataset import GoldRecord
from islands_harness.faults import FaultInjector
from islands_harness.loop import Prompts, RunRecord
from islands_harness.providers.base import ChatProvider, Message, Usage
from islands_harness.specs import RunSpec
from islands_harness.tools import Registry

# llama.cpp's reply when the model's own output fails the chat format's grammar, for
# example "The model produced output that does not match the expected peg-native format".
UNPARSEABLE_MARKERS = ("does not match the expected",)


def abort_cause(error: str) -> str:
    """``unparseable_model_output`` when the server rejected the model's own reply,
    ``transport`` otherwise. The first is a model failure that the transport layer reports as
    an abort: the verdicts keep the frozen rule (every aborted run leaves the denominators),
    and the snapshot also reports every result with these runs counted as failures (method
    note Section 7, decided before the first sweep run)."""
    return (
        "unparseable_model_output" if any(m in error for m in UNPARSEABLE_MARKERS) else "transport"
    )


class _AbortRecorder:
    """Forwards to the provider and keeps the text of the last TransportExhausted, so an
    aborted run can record why. The loop sees exactly what the provider returns or raises."""

    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider
        self.last_error: str | None = None

    async def complete(self, messages: Any, tools: Any, sampling: Any) -> Any:
        from islands_harness.providers.netguard import TransportExhausted

        try:
            return await self._provider.complete(messages, tools, sampling)
        except TransportExhausted as exc:
            self.last_error = str(exc)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


def _server_answered(record: Any) -> bool:
    """True when an aborted run's cause is the model's unparseable output: the server
    answered, so it is alive, and the abort is the model's failure, not the stack's."""
    return any(
        e.kind == "abort_cause" and e.data.get("cause") == "unparseable_model_output"
        for e in record.events
    )


class TransportHalt(RuntimeError):
    """Two runs in a row ended aborted_transport after their re-execution: the server or the
    API is down, not the runs. Neither is recorded; resume re-executes both."""


@dataclass
class RunSummary:
    executed: int = 0
    skipped_existing: int = 0
    aborted_transport: int = 0
    reexecuted: int = 0
    stopped_at_cap: bool = False
    spend_usd: float = 0.0
    by_outcome: dict[str, int] = field(default_factory=dict)


class Storage:
    """runs.jsonl plus sharded transcripts under one results directory."""

    def __init__(self, results_dir: Path) -> None:
        self.dir = Path(results_dir)
        self.runs_path = self.dir / "runs.jsonl"
        self.transcripts_dir = self.dir / "transcripts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def repair_tail(self) -> int:
        """Truncate a trailing partial line left by a crash mid-write and return the number of
        bytes removed. Without this, the next appended run would be glued to the fragment and
        the merged line would be unreadable. A complete file is left untouched."""
        if not self.runs_path.exists():
            return 0
        data = self.runs_path.read_bytes()
        if not data or data.endswith(b"\n"):
            return 0
        keep = data.rfind(b"\n") + 1
        with self.runs_path.open("r+b") as f:
            f.truncate(keep)
            f.flush()
            os.fsync(f.fileno())
        return len(data) - keep

    def existing_run_ids(self) -> set[str]:
        """Ids of every fully written run. A trailing partial line (no newline) is ignored."""
        ids: set[str] = set()
        if not self.runs_path.exists():
            return ids
        with self.runs_path.open("rb") as f:
            for raw in f:
                if not raw.endswith(b"\n"):
                    break
                line = raw.decode("utf-8").strip()
                if line:
                    ids.add(json.loads(line)["run_id"])
        return ids

    def append_run(self, record: RunRecord) -> None:
        """Append one line and fsync it. One os.write for the whole line."""
        line = (
            json.dumps(record_to_json(record), separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        fd = os.open(self.runs_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    def transcript_path(self, run_id: str) -> Path:
        """Short, sharded path: transcripts/<first two hex chars>/<run_id>.json."""
        return self.transcripts_dir / run_id[:2] / f"{run_id}.json"

    def write_transcript(
        self, run_id: str, transcript: list[Message], *, redact: bool = False
    ) -> Path:
        """Write the message list as JSON. With ``redact``, document text and tool payloads
        are replaced by ``{"sha256": ..., "length": ...}`` (client mode). Returns the path."""
        path = self.transcript_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [message_to_json(m, redact=redact) for m in transcript]
        path.write_text(
            json.dumps(
                {"run_id": run_id, "messages": rows}, indent=1, sort_keys=True, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path


def _redacted(text: str | None) -> Any:
    if text is None:
        return None
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "length": len(text)}


def message_to_json(m: Message, *, redact: bool = False) -> dict[str, Any]:
    """One transcript message. System and assistant text are kept even when redacting; the
    user message and tool payloads carry client data, so they are the redacted parts."""
    keep = m.role in ("system", "assistant") or not redact
    row: dict[str, Any] = {"role": m.role, "content": m.content if keep else _redacted(m.content)}
    if m.tool_calls:
        row["tool_calls"] = [
            {
                "id": c.id,
                "name": c.name,
                "arguments_text": c.arguments_text if not redact else _redacted(c.arguments_text),
                "malformed": c.malformed,
            }
            for c in m.tool_calls
        ]
    if m.tool_results:
        row["tool_results"] = [
            {
                "call_id": r.call_id,
                "name": r.name,
                "content": r.content if not redact else _redacted(r.content),
                "is_error": r.is_error,
                "executed": r.executed,
                "model_error": r.model_error,
                "fault": r.fault.to_json() if r.fault is not None else None,
            }
            for r in m.tool_results
        ]
    if m.raw_blocks:
        row["raw_blocks"] = m.raw_blocks
    return row


def message_from_json(row: dict[str, Any]) -> Message:
    """The inverse of message_to_json for an unredacted transcript. Parsed tool arguments and
    fault events are not restored: the canonical transcript hash and every comparison use the
    argument text and the result content, which are stored verbatim."""
    from islands_harness.providers.base import ToolCall, ToolResult

    calls = [
        ToolCall(
            id=str(c.get("id", "")),
            name=str(c["name"]),
            arguments=None,
            arguments_text=c["arguments_text"],
            malformed=bool(c.get("malformed", False)),
        )
        for c in row.get("tool_calls") or []
    ]
    results = [
        ToolResult(
            call_id=str(r.get("call_id", "")),
            name=str(r["name"]),
            content=r["content"],
            is_error=bool(r.get("is_error", False)),
            executed=bool(r.get("executed", False)),
            fault=None,
            model_error=bool(r.get("model_error", False)),
        )
        for r in row.get("tool_results") or []
    ]
    return Message(
        role=row["role"],
        content=row.get("content"),
        tool_calls=calls or None,
        tool_results=results or None,
        raw_blocks=row.get("raw_blocks"),
    )


def read_transcript(path: Path) -> list[Message]:
    """A stored transcript as the message list the loop produced."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [message_from_json(row) for row in data["messages"]]


def record_to_json(record: RunRecord) -> dict[str, Any]:
    """The runs.v1 row for a RunRecord: an explicit field-by-field mapping (never
    dataclasses.asdict, so a renamed field breaks a test rather than silently changing the
    schema); enums as their values."""
    u = record.usage
    return {
        "run_id": record.run_id,
        "model_id": record.model_id,
        "phase": record.phase,
        "mix": record.mix,
        "tools": record.tools,
        "fault_rate": record.fault_rate,
        "index": record.index,
        "doc_id": record.doc_id,
        "epoch": record.epoch,
        "sample_seed": record.sample_seed,
        "seed_offset": record.seed_offset,
        "tool_list": list(record.tool_list),
        "outcome": record.outcome.value,
        "turns": record.turns,
        "tool_calls": record.tool_calls,
        "seconds": record.seconds,
        "usage": {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_write_tokens": u.cache_write_tokens,
        },
        "cost_usd": record.cost_usd,
        "fault_events": [f.to_json() for f in record.fault_events],
        "first_fault_turn": record.first_fault_turn,
        "recovered": record.recovered,
        "submission": record.submission,
        "response_models": list(record.response_models),
        "fingerprints": list(record.fingerprints),
        "identity": record.identity,
        "transcript_path": record.transcript_path,
        "events": [{"kind": e.kind, "turn": e.turn, "data": e.data} for e in record.events],
        "score": record.score,
        "success": record.success,
        "per_field": record.per_field,
        "transport_retries": record.transport_retries,
        "malformed_calls": record.malformed_calls,
        "submission_attempted_in_text": record.submission_attempted_in_text,
    }


@dataclass
class SpendLedger:
    """Per-model spend against the dated price table and the cap."""

    prices: dict[str, float]  # USD per token: input, output, cache_read, cache_write
    cap_usd: float | None
    stop_at_fraction: float = 0.95
    spent_usd: float = 0.0

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.prices.get("input", 0.0)
            + usage.output_tokens * self.prices.get("output", 0.0)
            + usage.cache_read_tokens * self.prices.get("cache_read", 0.0)
            + usage.cache_write_tokens * self.prices.get("cache_write", 0.0)
        )

    def add(self, usage: Usage) -> float:
        c = self.cost(usage)
        self.spent_usd += c
        return c

    def would_exceed(self, forecast_usd: float) -> bool:
        """True when spent plus the forecast for the next run crosses stop_at_fraction * cap."""
        if self.cap_usd is None:
            return False
        return self.spent_usd + forecast_usd > self.stop_at_fraction * self.cap_usd


def grade_run(record: RunRecord, grader: Any, gold: GoldRecord, tau: float) -> RunRecord:
    """Fill score, success and per_field from the grader module's ``score``.

    A run without an accepted submission scores 0 (every field missing). ``success`` is
    ``score >= tau`` with tau from the frozen pre-registration; ``aborted_transport`` runs
    get ``success = None`` and are excluded downstream. The grader is the module object
    loaded from task.grader.
    """
    from islands_harness.loop import Outcome

    if record.outcome == Outcome.aborted_transport:
        record.score, record.success, record.per_field = None, None, None
        return record
    record.score, record.success, record.per_field = score_submission(
        grader, record.submission, gold.fields, tau
    )
    return record


def score_submission(
    grader: Any, submission: Any, gold_fields: dict[str, Any], tau: float
) -> tuple[float, bool, dict[str, float]]:
    """(score, success, per_field) for one submission, None meaning no accepted submission.
    The one scoring rule, shared by grade_run and ``verify --regrade``."""
    result = grader.score(submission, gold_fields)
    score = float(result.total)
    return score, score >= tau, dict(result.per_field)


@dataclass(frozen=True)
class TaskBundle:
    """Everything a run needs from the task, loaded once: tools, grader, prompt, documents,
    gold records and the dataset directory the tools read from."""

    registry: Registry
    grader: Any
    prompts: Prompts
    documents: dict[str, str]
    gold: dict[str, GoldRecord]
    dataset_dir: Path


def load_task(config: SnapshotConfig, repo_root: Path) -> TaskBundle:
    """Load the task named by the config. The system prompt is sent exactly as stored, with
    CRLF normalized to LF; nothing is added or stripped."""
    from islands_harness import dataset as ds
    from islands_harness.tools import load_module_from_path

    task_dir = Path(repo_root) / config.task.dir
    dataset_dir = task_dir / config.task.dataset
    registry = Registry.load(
        task_dir / "tools.py",
        required=config.task.required_tools,
        mixes=config.task.mixes,
        rung1=config.task.rung1,
    )
    system = (
        (task_dir / config.task.system_prompt).read_text(encoding="utf-8").replace("\r\n", "\n")
    )
    return TaskBundle(
        registry=registry,
        grader=load_module_from_path(task_dir / config.task.grader),
        prompts=Prompts(system=system),
        documents={d.id: d.text for d in ds.load(dataset_dir)},
        gold=ds.load_gold(dataset_dir),
        dataset_dir=dataset_dir,
    )


def load_prices(path: Path, model_name: str) -> dict[str, float]:
    """USD per token for ``model_name`` from a dated price table (configs/prices.v1.json).
    A model missing from the table is an error, never a zero price."""
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("unit") != "usd_per_token":
        raise ValueError(f"{path}: unit must be usd_per_token")
    try:
        entry = table["models"][model_name]
    except KeyError as exc:
        raise KeyError(f"{path} has no prices for {model_name!r}") from exc
    return {k: float(entry[k]) for k in ("input", "output", "cache_read", "cache_write")}


def round_robin(specs: list[RunSpec]) -> list[RunSpec]:
    """The i-th run of every cell before the (i+1)-th of any, so a sweep stopped early has
    partial coverage of every cell rather than complete coverage of a few."""
    return sorted(
        specs, key=lambda s: (s.index, s.cell.mix, s.cell.tools, s.cell.fault_rate, s.phase)
    )


async def execute_spec(
    spec: RunSpec,
    provider: ChatProvider,
    task: TaskBundle,
    *,
    config: SnapshotConfig,
    model: ModelConfig,
    agent: AgentConfig | None = None,
    injector: FaultInjector | None = None,
) -> tuple[RunRecord, list[Message]]:
    """Execute one spec exactly as the sweep does: the same run context, fault injector and
    per-run sampling. The sweep, ``replay`` and ``verify --reexecute`` all call this, so a
    replayed run cannot differ from the original through the way it was started. A run that
    ends ``aborted_transport`` gains an ``abort_cause`` event with the provider's last error
    and its cause (``abort_cause``); nothing else about any run changes."""
    from islands_harness.loop import Event, Outcome, run_one
    from islands_harness.providers.factory import sampling_for
    from islands_harness.tools import RunContext

    ctx = RunContext(
        doc_id=spec.doc_id,
        document_text=task.documents[spec.doc_id],
        dataset_dir=task.dataset_dir,
        tools={},
    )
    recorder = _AbortRecorder(provider)
    record, transcript = await run_one(
        spec,
        recorder,  # type: ignore[arg-type]
        task.registry,
        injector or FaultInjector(config.faults, spec, config.snapshot.root_seed),
        agent or config.agent,
        task.prompts,
        ctx=ctx,
        sampling=sampling_for(model, spec.sample_seed),
    )
    if record.outcome == Outcome.aborted_transport and recorder.last_error is not None:
        error = recorder.last_error
        record.events.append(
            Event("abort_cause", record.turns, {"cause": abort_cause(error), "error": error[:600]})
        )
    return record, transcript


async def run_specs(
    specs: list[RunSpec],
    provider: ChatProvider,
    task: TaskBundle,
    storage: Storage,
    ledger: SpendLedger | None,
    *,
    config: SnapshotConfig,
    model: ModelConfig,
    tau: float,
    agent: AgentConfig | None = None,
    concurrency: int | None = None,
    order: str | None = None,
    resume: bool = True,
    progress: Callable[[str], None] | None = None,
    injector_factory: Callable[[RunSpec], FaultInjector] | None = None,
) -> RunSummary:
    """Drive the specs to completion under a semaphore and return the summary.

    Order: ``round_robin`` (the default from the config) or ``sequential``. Resume: specs whose
    run id is already in runs.jsonl are skipped; a partial trailing line from a crash is
    truncated first. Without ``resume`` an existing runs.jsonl is refused rather than appended
    to. Spend: before each run starts, the ledger is asked whether the running mean cost per
    run would cross 95 percent of the cap; if so no further run starts and ``stopped_at_cap``
    is set. A run that ends ``aborted_transport`` is re-executed once; a second abort is kept
    with that outcome, written when the next run completes (or at the end of the pass). Two
    such runs in a row raise ``TransportHalt`` and neither is written. Both attempts are
    billed to the ledger; the row keeps only the second attempt's cost. Progress lines report counts and spend, never a rate or a verdict.
    Any other exception is a harness bug and stops the sweep; resume picks up afterwards.
    """
    from islands_harness.loop import Outcome

    agent = agent or config.agent
    concurrency = concurrency or model.concurrency
    order = order or config.sweep.order
    if injector_factory is None:

        def injector_factory(spec: RunSpec) -> FaultInjector:
            return FaultInjector(config.faults, spec, config.snapshot.root_seed)

    summary = RunSummary()
    storage.repair_tail()
    existing = storage.existing_run_ids()
    if existing and not resume:
        raise FileExistsError(
            f"{storage.runs_path} already holds runs; resume, or choose a fresh directory"
        )
    todo = [s for s in specs if s.run_id not in existing]
    summary.skipped_existing = len(specs) - len(todo)
    ordered = round_robin(todo) if order == "round_robin" else list(todo)
    semaphore = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    held: list[tuple[RunSpec, RunRecord, list[Message]]] = []  # an abort awaiting the next run

    async def execute(spec: RunSpec) -> tuple[RunRecord, list[Message]]:
        return await execute_spec(
            spec,
            provider,
            task,
            config=config,
            model=model,
            agent=agent,
            injector=injector_factory(spec),
        )

    async def worker(spec: RunSpec) -> None:
        async with semaphore:
            if stop.is_set():
                return
            mean_cost = summary.spend_usd / summary.executed if summary.executed else 0.0
            if ledger is not None and ledger.would_exceed(mean_cost):
                summary.stopped_at_cap = True
                stop.set()
                return
            record, transcript = await execute(spec)
            discarded = None  # usage of an aborted first attempt: billed, though not kept
            if record.outcome == Outcome.aborted_transport:
                summary.reexecuted += 1
                discarded = record.usage
                record, transcript = await execute(spec)
            if ledger is not None:
                if discarded is not None:
                    ledger.add(discarded)
                record.cost_usd = ledger.add(record.usage)
                summary.spend_usd = ledger.spent_usd
            if record.outcome == Outcome.aborted_transport and not _server_answered(record):
                if held:
                    stop.set()
                    raise TransportHalt(
                        f"runs {held[0][0].run_id} and {spec.run_id} both ended aborted_transport "
                        "twice in a row; the server looks down. Neither was recorded, and resume "
                        "re-executes them"
                    )
                held.append((spec, record, transcript))
                return
            while held:
                record_run(*held.pop())
            record_run(spec, record, transcript)

    def record_run(spec: RunSpec, record: RunRecord, transcript: list[Message]) -> None:
        if record.outcome == Outcome.aborted_transport:
            summary.aborted_transport += 1
        record.transcript_path = (
            storage.write_transcript(spec.run_id, transcript).relative_to(storage.dir).as_posix()
        )
        grade_run(record, task.grader, task.gold[spec.doc_id], tau)
        storage.append_run(record)
        summary.executed += 1
        summary.by_outcome[record.outcome.value] = (
            summary.by_outcome.get(record.outcome.value, 0) + 1
        )
        if progress is not None:
            spend = f", spend ${summary.spend_usd:.2f}" if ledger is not None else ""
            progress(f"{summary.executed + summary.skipped_existing}/{len(specs)} runs{spend}")

    tasks = [asyncio.create_task(worker(s)) for s in ordered]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # One failure (or the sweep itself being cancelled) stops every worker. A worker
        # cancelled mid-run has written nothing, because the write is the last, synchronous
        # step, so its run is simply re-executed on resume.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    while held:  # an abort at the end of the pass, with no later run to wait for
        record_run(*held.pop())
    return summary
