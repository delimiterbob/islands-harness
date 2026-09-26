"""The fault injector: the one wrapper around ``tools.execute``.

On every tool call, with probability equal to the cell's fault rate, the injector returns a
fault instead of (or on top of) the genuine result. The decision and the kind are pure
functions of the run's identity and the call index (ARCHITECTURE.md Section 7):

    u1 = unit("fault", root_seed, model_id, mix, doc_id, epoch, call_index);  fires iff u1 < rate
    u2 = unit("kind",  same indices) mapped through the cumulative kind weights

Nesting across rates. The scope contains neither the tool count nor the fault rate, so the
same call index in the same (model, mix, document, epoch) has the same u1 in every cell: a
call faulted at 10 percent is faulted at 20 through 50 percent, and at rate 0 nothing ever
fires. These common random numbers lower the variance of every contrast between cells and
correlate cells, which the cluster-robust covariance by document absorbs in the fit. The
tool count still changes which calls the model makes, so call index i is not the same call
across tool columns; nesting is exact along the fault axis, not the tool axis.

Kinds and the frozen result strings (hashed with this file by freeze):

    timeout          not executed;  TIMEOUT_RESULT (JSON error naming the tool, 30 s)
    error_response   not executed;  ERROR_RESULT (JSON error, status 503)
    empty_result     not executed;  EMPTY_RESULT (the empty string)
    garbled_payload  executed;      garble(genuine result)

Because the three unexecuted kinds never reach the tool, a faulted ``submit_record`` stores
nothing and the model must resubmit; the garbled kind stores the record and garbles the
acknowledgement only. The ``executed`` flag on every FaultEvent records which happened.

Timeouts do not sleep: turns are the recovery unit, and real sleeps would multiply hosted
cost (D16). ``simulate_delay_s`` exists for a client who wants wall-clock realism.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from enum import StrEnum

from islands_harness import tools
from islands_harness.config import FaultConfig
from islands_harness.providers.base import ToolCall, ToolResult
from islands_harness.rng import choice_weighted, unit
from islands_harness.specs import RunSpec


class FaultKind(StrEnum):
    timeout = "timeout"
    garbled_payload = "garbled_payload"
    error_response = "error_response"
    empty_result = "empty_result"


@dataclass(frozen=True)
class FaultEvent:
    """One injected fault, as logged in runs.jsonl and the transcript."""

    turn: int
    call_index: int
    tool: str
    args_sha256: str
    kind: FaultKind
    executed: bool

    def to_json(self) -> dict[str, object]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


def timeout_result(tool: str, seconds: int = 30) -> str:
    return json.dumps({"error": "timeout", "tool": tool, "timeout_s": seconds}, sort_keys=True)


TIMEOUT_RESULT_TEMPLATE = timeout_result("{tool}")
ERROR_RESULT = json.dumps({"error": "service_unavailable", "status": 503}, sort_keys=True)
EMPTY_RESULT = ""

# Fixed alphabet for garbling: printable ASCII that is common in JSON and CSV payloads, so a
# garbled result looks like corruption, not like a marker.
GARBLE_ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]",:;.-_ #'


class FaultInjector:
    """Wraps ``tools.execute`` for one run."""

    def __init__(self, config: FaultConfig, spec: RunSpec, root_seed: int) -> None:
        self.config = config
        self.spec = spec
        self.root_seed = root_seed
        self.rate = spec.cell.fault_rate
        self.events: list[FaultEvent] = []

    # -- decision ----------------------------------------------------------------------

    def _indices(self, call_index: int) -> tuple[object, ...]:
        return (
            self.root_seed,
            self.spec.model_id,
            self.spec.cell.mix,
            self.spec.doc_id,
            self.spec.epoch,
            call_index,
        )

    def decide(self, call_index: int, tool: str) -> FaultKind | None:
        """The kind to inject at this call index, or None. Pure."""
        if self.rate <= 0.0:
            return None
        if self.config.applies_to == "exempt_submit" and tool == "submit_record":
            return None
        u1 = unit("fault", *self._indices(call_index))
        if u1 >= self.rate:
            return None
        u2 = unit("kind", *self._indices(call_index))
        return FaultKind(choice_weighted(u2, list(self.config.kinds), list(self.config.weights)))

    # -- execution ---------------------------------------------------------------------

    async def call(self, tool_call: ToolCall, ctx: tools.RunContext) -> ToolResult:
        """Execute or fault one tool call. The only caller is loop.run_one."""
        kind = self.decide(ctx.call_index, tool_call.name)
        if kind is None:
            return await tools.execute(tool_call, ctx)

        if self.config.simulate_delay_s > 0 and kind == FaultKind.timeout:
            await asyncio.sleep(self.config.simulate_delay_s)

        executed = kind == FaultKind.garbled_payload
        event = FaultEvent(
            turn=ctx.turn,
            call_index=ctx.call_index,
            tool=tool_call.name,
            args_sha256=tools.args_sha256(
                tool_call.arguments if tool_call.arguments is not None else tool_call.arguments_text
            ),
            kind=kind,
            executed=executed,
        )
        self.events.append(event)

        if kind == FaultKind.timeout:
            content = timeout_result(tool_call.name)
        elif kind == FaultKind.error_response:
            content = ERROR_RESULT
        elif kind == FaultKind.empty_result:
            content = EMPTY_RESULT
        else:
            genuine = await tools.execute(tool_call, ctx)
            content = garble(genuine.content, self.config, self._indices(ctx.call_index))
        return ToolResult(
            call_id=tool_call.id,
            name=tool_call.name,
            content=content,
            is_error=False,  # a fault is delivered as an ordinary result; the model must notice
            executed=executed,
            fault=event,
        )


def garble(text: str, config: FaultConfig, indices: tuple[object, ...]) -> str:
    """Corrupt a genuine result deterministically.

    Replace ``config.garble.char_fraction`` of the positions (chosen by
    ``unit("garble", *indices, k)`` for k over positions, kept when u < char_fraction) with a
    character from GARBLE_ALPHABET selected by ``unit("garble", *indices, "sub", k)``, never
    the original character at that position. Then, if ``config.garble.truncate``, cut the
    string at position ``floor(len * (0.3 + 0.7 * unit("garble", *indices, "cut")))``.
    An empty or single-character input is returned as a single replaced character.

    Property (tests/test_harness.py): the output never parses as JSON and never equals the
    original. Both are enforced structurally: at least one replacement always happens, and
    then, while the output parses as JSON or equals the original, a quote character is
    appended. A quote after a complete JSON value is never valid JSON, so this ends within
    two appends. The check applies to every input, not only to JSON originals, because a
    truncated CSV or plain-text result can parse by chance.
    """

    def sub(c: str, k: int) -> str:
        i = int(unit("garble", *indices, "sub", k) * len(GARBLE_ALPHABET)) % len(GARBLE_ALPHABET)
        return (
            GARBLE_ALPHABET[i]
            if GARBLE_ALPHABET[i] != c
            else GARBLE_ALPHABET[(i + 1) % len(GARBLE_ALPHABET)]
        )

    chars = [sub(text[:1], 0)] if len(text) <= 1 else list(text)
    if len(text) > 1:
        hits = [
            k
            for k in range(len(chars))
            if unit("garble", *indices, k) < config.garble.char_fraction
        ] or [0]
        for k in hits:
            chars[k] = sub(chars[k], k)
        if config.garble.truncate:
            cut = int(len(chars) * (0.3 + 0.7 * unit("garble", *indices, "cut")))
            chars = chars[: max(cut, 1)]
    out = "".join(chars)
    while _parses(out) or out == text:
        out += '"'
    return out


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return True
