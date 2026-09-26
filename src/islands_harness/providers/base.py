"""Provider protocol, provider-neutral message model, sampling record, capability table and
the scripted fake provider.

The loop speaks only this model. Each adapter translates it to one wire format and back,
keeps the raw request and response bodies for the transcript, and applies the retry policy
from netguard. What a provider cannot accept is declared in CAPABILITIES rather than
silently dropped: the adapter raises on a refused field so a config that asks for
``temperature`` on Anthropic fails at doctor time, not mid-sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from islands_harness.faults import FaultEvent

Role = Literal["system", "user", "assistant", "tool"]


class StopReason(StrEnum):
    end = "end"
    tool_calls = "tool_calls"
    max_tokens = "max_tokens"
    refusal = "refusal"
    other = "other"


@dataclass(frozen=True)
class ToolCall:
    """A tool call as emitted by the model. ``arguments`` is None when the argument text was
    not valid JSON; ``arguments_text`` always holds what the model actually wrote."""

    id: str
    name: str
    arguments: dict[str, Any] | None
    arguments_text: str
    malformed: bool = False


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool
    executed: bool
    fault: FaultEvent | None = None
    model_error: bool = False  # unknown tool or malformed arguments


@dataclass(frozen=True)
class Usage:
    """Token counts for one call or one run, in one convention across providers:
    ``input_tokens`` is uncached input only; cached input is ``cache_read_tokens`` and input
    written to a cache is ``cache_write_tokens``. Each count is priced once, at its own rate."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        """Every token processed, cached or not; the agent's token_limit is checked on this."""
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.output_tokens
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class Message:
    """One transcript entry. ``raw_blocks`` keeps provider-native content (for Anthropic, the
    thinking and tool_use blocks exactly as returned) so the adapter can replay it unchanged."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    raw_blocks: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class Turn:
    """What one model call returned."""

    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: StopReason
    identity: dict[str, Any]  # served model id, fingerprint, response id, request dates
    message: Message  # the assistant message to append, raw blocks included
    raw_request: dict[str, Any]
    raw_response: dict[str, Any]
    transport_retries: int = 0


@dataclass(frozen=True)
class Sampling:
    """The per-run sampling configuration as sent. Built from config.SamplingConfig with the
    per-run seed substituted for ``per_run``."""

    temperature: float | None
    top_p: float | None
    max_tokens: int
    seed: int | None
    thinking: Literal["adaptive", "disabled"] | None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None


# Which Sampling fields each provider accepts. Anything else is refused at construction.
CAPABILITIES: dict[str, dict[str, frozenset[str]]] = {
    "openai_compat": {
        "accepts": frozenset({"temperature", "top_p", "max_tokens", "seed"}),
        "refuses": frozenset({"thinking", "effort"}),
    },
    "anthropic": {
        # Current Claude models reject temperature/top_p/top_k (D4); no seed exists.
        "accepts": frozenset({"max_tokens", "thinking", "effort"}),
        "refuses": frozenset({"temperature", "top_p", "seed"}),
    },
}


def check_capabilities(provider: str, sampling: Sampling) -> None:
    """Raise ValueError if ``sampling`` sets a field the provider refuses."""
    refused = CAPABILITIES[provider]["refuses"]
    offending = [f for f in refused if getattr(sampling, f) is not None]
    if offending:
        raise ValueError(f"provider {provider!r} refuses sampling fields {offending}")


class ContextOverflow(Exception):
    """The provider reported that the request exceeds the model's context."""


class ChatProvider(Protocol):
    async def complete(self, messages: list[Message], tools: list[Any], sampling: Sampling) -> Turn:
        """One model call. Retries transport failures per netguard.RetryPolicy and raises
        TransportExhausted when they run out; never retries anything else."""
        ...

    def identity(self) -> dict[str, Any]:
        """Static identity: configured model id, base URL host, transport, server version."""
        ...


@dataclass(frozen=True)
class ScriptedTurn:
    """One scripted model response for the fake provider."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason = StopReason.end
    usage: Usage = field(default_factory=lambda: Usage(100, 20))
    raise_transport: bool = False  # simulate a retryable transport failure on this call
    raise_overflow: bool = False


class FakeProvider:
    """Replays scripted turns in order; used by tests and ``islands selftest``.

    A script that runs out raises, so a loop that makes more calls than scripted is a test
    failure rather than a silent extra turn. Requests are recorded on ``calls`` so tests can
    assert the loop added nothing of its own.
    """

    def __init__(self, script: list[ScriptedTurn], *, model_id: str = "fake") -> None:
        self.script = list(script)
        self.model_id = model_id
        self.calls: list[tuple[list[Message], list[Any], Sampling]] = []

    async def complete(self, messages: list[Message], tools: list[Any], sampling: Sampling) -> Turn:
        from islands_harness.providers.netguard import TransportExhausted

        if not self.script:
            raise AssertionError("FakeProvider script exhausted")
        self.calls.append(
            (
                [
                    Message(m.role, m.content, m.tool_calls, m.tool_results, m.raw_blocks)
                    for m in messages
                ],
                list(tools),
                sampling,
            )
        )
        step = self.script.pop(0)
        if step.raise_transport:
            raise TransportExhausted("scripted transport failure")
        if step.raise_overflow:
            raise ContextOverflow("scripted context overflow")
        message = Message(
            role="assistant", content=step.text or None, tool_calls=list(step.tool_calls) or None
        )
        return Turn(
            text=step.text,
            tool_calls=list(step.tool_calls),
            usage=step.usage,
            stop_reason=(
                step.stop_reason
                if not step.tool_calls or step.stop_reason == StopReason.max_tokens
                else StopReason.tool_calls
            ),
            identity={
                "model": self.model_id,
                "fingerprint": "fake",
                "response_id": f"fake-{len(self.calls)}",
            },
            message=message,
            raw_request={
                "messages": len(messages),
                "tools": [t.name if hasattr(t, "name") else t for t in tools],
            },
            raw_response={"scripted": True},
        )

    def identity(self) -> dict[str, Any]:
        return {"model": self.model_id, "provider": "fake"}
