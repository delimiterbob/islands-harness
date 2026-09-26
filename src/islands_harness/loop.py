"""The agent loop: the only place the model and the tools meet.

One run is: the system prompt, a user message naming the document id, then at most
``agent.max_turns`` iterations of model call, sequential tool execution in emitted order,
append. It ends at the first accepted submit, at a model stop without a submit, at a limit,
at a refusal, or at context overflow (ARCHITECTURE.md Section 3, D1). A turn cut off at
``max_tokens`` ends the run as ``limit_output`` and nothing in it is executed, because its
tool arguments may be partial.

Invariants, each one a test in tests/test_harness.py:

    1. The loop adds no text of its own. The transcript contains only the system prompt, the
       user message, the model's turns and the tool results. No nudges, no "continue", no
       summaries, no repaired arguments.
    2. Parallel tool calls in one assistant turn are executed sequentially in emitted order,
       each with its own call index, and all results go back in one message.
    3. A tool is never retried by the loop. A fault reaches the model unmodified.
    4. Malformed arguments get the frozen catalog error from tools.execute and nothing else.
    5. The only path to a tool implementation is injector.call -> tools.execute.
    6. Transport failures are handled inside the provider (netguard.with_retry); if they are
       exhausted the run ends as ``aborted_transport`` and is excluded from every denominator.

Target size: about 150 lines including this docstring, so one reviewer reads the whole
instrument in one sitting. If it grows past that, something has moved in that belongs in a
provider or in faults.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from islands_harness.config import AgentConfig
from islands_harness.faults import FaultEvent, FaultInjector
from islands_harness.providers.base import (
    ChatProvider,
    ContextOverflow,
    Message,
    Sampling,
    StopReason,
    ToolResult,
    Usage,
)
from islands_harness.providers.netguard import TransportExhausted
from islands_harness.specs import RunSpec
from islands_harness.tools import MALFORMED_ARGUMENTS_TEXT, Registry, RunContext


class Outcome(StrEnum):
    submitted = "submitted"
    no_submission = "no_submission"
    limit_turns = "limit_turns"
    limit_tool_calls = "limit_tool_calls"
    limit_tokens = "limit_tokens"
    limit_time = "limit_time"
    limit_output = "limit_output"  # a turn cut off at max_tokens
    refusal = "refusal"
    context_overflow = "context_overflow"
    aborted_transport = "aborted_transport"


@dataclass(frozen=True)
class Event:
    """One line of the run's event list: model_turn, tool_call, fault, transport_retry, limit."""

    kind: str
    turn: int
    data: dict[str, Any]


@dataclass(frozen=True)
class Prompts:
    system: str  # the hashed system prompt bytes, decoded
    user_template: str = "Document id: {doc_id}"  # frozen; contains no date, no run id


@dataclass
class RunRecord:
    """One line of runs.jsonl. Grading fields are filled by runner.grade_run."""

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
    outcome: Outcome
    turns: int
    tool_calls: int
    seconds: float
    usage: Usage
    cost_usd: float | None
    fault_events: list[FaultEvent]
    first_fault_turn: int | None
    recovered: bool | None  # faulted and finished with an accepted submit
    submission: dict[str, Any] | None
    response_models: list[str]
    fingerprints: list[str]
    identity: dict[str, Any]
    transcript_path: str
    events: list[Event] = field(default_factory=list)
    score: float | None = None
    success: bool | None = None
    per_field: dict[str, float] | None = None
    transport_retries: int = 0
    malformed_calls: int = 0
    submission_attempted_in_text: bool | None = None  # rung 1 only


async def run_one(
    spec: RunSpec,
    provider: ChatProvider,
    registry: Registry,
    injector: FaultInjector,
    agent: AgentConfig,
    prompts: Prompts,
    *,
    ctx: RunContext,
    sampling: Sampling,
) -> tuple[RunRecord, list[Message]]:
    """Execute one run and return its record and its transcript (the message list).

    The order of checks inside a turn is fixed: transport and overflow, refusal, output limit,
    no tool calls, then each call in emitted order (the tool-call limit is checked before each call),
    then acceptance, the token limit and the time limit. Rung 1 (no submit tool offered) also
    sets ``submission_attempted_in_text``; that flag is reported and never enters a statistic.
    """
    messages = [
        Message("system", prompts.system),
        Message("user", prompts.user_template.format(doc_id=spec.doc_id)),
    ]
    offered = registry.for_rung(spec.cell.mix, spec.cell.tools)
    ctx.tools = {t.name: t for t in offered}
    events: list[Event] = []
    usage, retries, malformed, models, prints = Usage(), 0, 0, [], []
    started = time.monotonic()
    outcome: Outcome | None = None
    for turn in range(agent.max_turns):
        ctx.turn = turn
        try:
            result = await provider.complete(messages, offered, sampling)
        except TransportExhausted:
            outcome = Outcome.aborted_transport
            break
        except ContextOverflow:
            outcome = Outcome.context_overflow
            break
        messages.append(result.message)
        usage, retries = usage + result.usage, retries + result.transport_retries
        for key, bucket in (("model", models), ("fingerprint", prints)):
            if result.identity.get(key) and result.identity[key] not in bucket:
                bucket.append(result.identity[key])
        events.append(
            Event(
                "model_turn",
                turn,
                {
                    "stop_reason": result.stop_reason.value,
                    "calls": len(result.tool_calls),
                    "usage": vars(result.usage),
                    "identity": result.identity,
                },
            )
        )
        if result.stop_reason == StopReason.refusal:
            outcome = Outcome.refusal
            break
        if result.stop_reason == StopReason.max_tokens:
            outcome = Outcome.limit_output  # truncated: its tool calls are never executed
            break
        if not result.tool_calls:
            outcome = Outcome.no_submission
            break
        results: list[ToolResult] = []
        accepted = False
        for call in result.tool_calls:
            if ctx.call_index >= agent.max_tool_calls:
                outcome = Outcome.limit_tool_calls
                break
            tool_result = await injector.call(call, ctx)
            events.append(
                Event(
                    "tool_call",
                    turn,
                    {
                        "call_index": ctx.call_index,
                        "name": call.name,
                        "executed": tool_result.executed,
                        "model_error": tool_result.model_error,
                    },
                )
            )
            if tool_result.fault is not None:
                events.append(Event("fault", turn, tool_result.fault.to_json()))
            malformed += tool_result.content == MALFORMED_ARGUMENTS_TEXT
            ctx.call_index += 1
            results.append(tool_result)
            accepted = accepted or (
                call.name == SUBMIT_TOOL and tool_result.executed and ctx.submission_accepted
            )
        if outcome is not None:
            break
        messages.append(Message("tool", tool_results=results))
        if accepted:
            outcome = Outcome.submitted
        elif usage.total > agent.token_limit:
            outcome = Outcome.limit_tokens
        elif time.monotonic() - started > agent.time_limit_s:
            outcome = Outcome.limit_time
        if outcome is not None:
            break
    if outcome is None:
        outcome = Outcome.limit_turns
    faults = list(injector.events)
    record = RunRecord(
        run_id=spec.run_id,
        model_id=spec.model_id,
        phase=spec.phase,
        mix=spec.cell.mix,
        tools=spec.cell.tools,
        fault_rate=spec.cell.fault_rate,
        index=spec.index,
        doc_id=spec.doc_id,
        epoch=spec.epoch,
        sample_seed=spec.sample_seed,
        seed_offset=spec.seed_offset,
        tool_list=list(spec.tool_list),
        outcome=outcome,
        turns=sum(1 for m in messages if m.role == "assistant"),
        tool_calls=ctx.call_index,
        seconds=round(time.monotonic() - started, 3),
        usage=usage,
        cost_usd=None,
        fault_events=faults,
        first_fault_turn=min((f.turn for f in faults), default=None),
        recovered=(outcome == Outcome.submitted) if faults else None,
        submission=ctx.submission,
        response_models=models,
        fingerprints=prints,
        identity=provider.identity(),
        transcript_path="",
        events=events,
        transport_retries=retries,
        malformed_calls=int(malformed),
        submission_attempted_in_text=_text_submission(messages) if spec.cell.tools == 1 else None,
    )
    return record, messages


SUBMIT_TOOL = "submit_record"


def _text_submission(messages: list[Message]) -> bool:
    """True if any assistant text holds a JSON object with an ``invoice_number`` key."""
    decoder = json.JSONDecoder()
    for m in messages:
        text = (m.content or "") if m.role == "assistant" else ""
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "invoice_number" in obj:
                return True
    return False
