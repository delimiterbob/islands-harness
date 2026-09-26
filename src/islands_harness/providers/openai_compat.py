"""Chat Completions adapter over httpx2 for llama.cpp's llama-server, OpenRouter, OpenAI and Azure OpenAI.

The harness writes the wire format itself (D3): the request body is built here, logged
verbatim in the transcript, sent through the allowlist transport, and parsed here. No SDK,
so nothing is retried, reshaped, or validated by a library between the model and the loop.
Retries are the harness's own (netguard.with_retry) and are logged as transport_retry events.

Request body (Chat Completions):

    {"model", "messages", "tools": [{"type": "function", "function": {name, description,
      parameters}}], "tool_choice": "auto", "parallel_tool_calls": true|false,
      "temperature", "top_p", "seed", "max_tokens", **extra_body}

``strict`` is never set on a function schema (D11). ``extra_body`` carries server-specific
fields. For llama-server: the explicit sampler chain (``samplers``, ``top_k``, ``min_p``),
``reasoning_effort``, ``chat_template_kwargs``, ``cache_prompt: false`` and
``parallel_tool_calls``. llama-server returns reasoning in ``message.reasoning_content``,
which the transcript keeps.

Response fields kept: ``choices[0].message.content``, ``choices[0].message.tool_calls``
(id, function.name, function.arguments as text; parsed here, malformed JSON flagged),
``choices[0].finish_reason`` (stop -> end, tool_calls -> tool_calls, length -> max_tokens,
content_filter -> refusal, other -> other), ``usage`` (prompt_tokens, completion_tokens,
prompt_tokens_details.cached_tokens when present), ``model``, ``system_fingerprint``, ``id``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx2 as httpx

from islands_harness.providers.base import (
    ContextOverflow,
    Message,
    Sampling,
    StopReason,
    ToolCall,
    Turn,
    Usage,
    check_capabilities,
)
from islands_harness.providers.netguard import (
    RetryPolicy,
    TransportError,
    build_http_client,
    with_retry,
)
from islands_harness.tools import ToolSpec


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str | None,
        sampling: Sampling,
        extra_body: dict[str, Any] | None,
        allowed_hosts: list[str],
        timeout_s: float = 600.0,
        parallel_tool_calls: bool = True,
        retry_policy: RetryPolicy = RetryPolicy(),
        log: Any = None,
    ) -> None:
        check_capabilities("openai_compat", sampling)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.sampling = sampling
        self.extra_body = dict(extra_body or {})
        self.parallel_tool_calls = parallel_tool_calls
        self.retry_policy = retry_policy
        self.log = log or (lambda event: None)
        self.client: httpx.AsyncClient = build_http_client(allowed_hosts, timeout_s)
        self._server_version: str | None = None

    # -- translation -------------------------------------------------------------------

    def build_body(
        self, messages: list[Message], tools: list[ToolSpec], sampling: Sampling
    ) -> dict[str, Any]:
        """The exact request body. Pure, so a test can assert it and the transcript keeps it."""
        wire: list[dict[str, Any]] = []
        for m in messages:
            converted = self._to_wire(m)
            wire.extend(converted if isinstance(converted, list) else [converted])
        body: dict[str, Any] = {
            "model": self.model,
            "messages": wire,
            "max_tokens": sampling.max_tokens,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t.schema()} for t in tools]
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = self.parallel_tool_calls
        if sampling.temperature is not None:
            body["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            body["top_p"] = sampling.top_p
        if sampling.seed is not None:
            body["seed"] = wire_seed(sampling.seed)
        body.update(self.extra_body)
        return body

    @staticmethod
    def _to_wire(m: Message) -> dict[str, Any] | list[dict[str, Any]]:
        """Provider-neutral Message -> Chat Completions message(s).

        Tool results become one ``role: tool`` message per result (the wire format has no
        multi-result message); the loop still appends them as one Message, which is the unit
        the transcript and the invariants speak about.

        A tool Message with several results returns a list; callers flatten. Assistant tool
        calls are replayed with the argument text exactly as the model wrote it, never a
        re-serialization. Reasoning returned as ``reasoning_content`` (kept in ``raw_blocks``)
        is sent back verbatim on later turns: gpt-oss expects its prior reasoning within a
        tool-calling chain, and replaying the model's own text adds nothing to the transcript.
        """
        if m.role in ("system", "user"):
            return {"role": m.role, "content": m.content or ""}
        if m.role == "assistant":
            # Text if the model wrote any; null for a tool-calls-only turn; otherwise "".
            content = m.content if m.content else (None if m.tool_calls else "")
            out: dict[str, Any] = {"role": "assistant", "content": content}
            if m.tool_calls:
                out["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments_text},
                    }
                    for c in m.tool_calls
                ]
            reasoning = [
                b["text"] for b in (m.raw_blocks or []) if b.get("type") == "reasoning_content"
            ]
            if reasoning:
                out["reasoning_content"] = reasoning[0]
            return out
        return [
            {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
            for r in (m.tool_results or [])
        ]

    @staticmethod
    def parse_turn(body: dict[str, Any], request: dict[str, Any]) -> Turn:
        """Chat Completions response -> Turn. Malformed tool-call argument JSON yields a
        ToolCall with ``arguments=None, malformed=True``; the raw text is kept. A response
        with tool calls is a tool_calls stop whatever finish_reason says, because servers
        disagree on that field when the model both writes text and calls a tool; the one
        exception is ``length``, which always maps to max_tokens so a truncated call is never
        executed.
        """
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        calls = [_tool_call(tc) for tc in (msg.get("tool_calls") or [])]
        reasoning = msg.get("reasoning_content")
        finish = choice.get("finish_reason")
        if finish == "length":
            stop = StopReason.max_tokens
        else:
            stop = StopReason.tool_calls if calls else _finish_reason(finish)
        message = Message(
            role="assistant",
            content=text or None,
            tool_calls=calls or None,
            raw_blocks=[{"type": "reasoning_content", "text": reasoning}] if reasoning else None,
        )
        identity = {
            "model": body.get("model"),
            "fingerprint": body.get("system_fingerprint"),
            "response_id": body.get("id"),
        }
        return Turn(
            text=text,
            tool_calls=calls,
            usage=_usage(body.get("usage")),
            stop_reason=stop,
            identity=identity,
            message=message,
            raw_request=request,
            raw_response=body,
        )

    # -- transport ---------------------------------------------------------------------

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransportError(str(exc), retryable=True) from exc
        if (
            response.status_code == 400
            and b"context" in response.content.lower()
            and b"length" in response.content.lower()
        ):
            raise ContextOverflow(response.text[:500])
        if response.status_code >= 400:
            from islands_harness.providers.netguard import classify_status

            raise TransportError(
                f"HTTP {response.status_code}: {response.text[:300]}",
                retryable=classify_status(response.status_code),
                status=response.status_code,
            )
        self._server_version = response.headers.get("server") or self._server_version
        return response.json()

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec], sampling: Sampling
    ) -> Turn:
        body = self.build_body(messages, tools, sampling)
        retries = {"n": 0}

        def log(event: dict[str, Any]) -> None:
            retries["n"] += 1
            self.log(event)

        sent_at = datetime.now(UTC).isoformat()
        raw = await with_retry(lambda: self._post(body), self.retry_policy, log)
        turn = self.parse_turn(raw, body)
        identity = dict(turn.identity)
        identity["request_date"] = sent_at
        return Turn(
            text=turn.text,
            tool_calls=turn.tool_calls,
            usage=turn.usage,
            stop_reason=turn.stop_reason,
            identity=identity,
            message=turn.message,
            raw_request=body,
            raw_response=raw,
            transport_retries=retries["n"],
        )

    def identity(self) -> dict[str, Any]:
        return {
            "provider": "openai_compat",
            "model": self.model,
            "base_url": self.base_url,
            "server": self._server_version,
        }

    async def aclose(self) -> None:
        await self.client.aclose()


LLAMA_RANDOM_SEED = 0xFFFFFFFF  # llama-server reads this seed value as "pick a random seed"


def wire_seed(seed: int) -> int:
    """The seed as sent: the low 32 bits of the 63-bit run seed, because llama-server's seed
    is a uint32. The one value llama-server treats as "random" is stepped down by one so a
    run can never silently become nondeterministic. OpenAI-compatible hosts accept any int, so
    the same value is sent everywhere; the request body in the transcript records it."""
    low = seed & 0xFFFFFFFF
    return low - 1 if low == LLAMA_RANDOM_SEED else low


def _finish_reason(value: str | None) -> StopReason:
    return {
        "stop": StopReason.end,
        "tool_calls": StopReason.tool_calls,
        "length": StopReason.max_tokens,
        "content_filter": StopReason.refusal,
    }.get(value or "", StopReason.other)


def _usage(u: dict[str, Any] | None) -> Usage:
    """Chat Completions ``prompt_tokens`` includes cached tokens; the harness convention
    (base.Usage) counts uncached input separately, so cached tokens are subtracted once here."""
    u = u or {}
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
    prompt = int(u.get("prompt_tokens", 0))
    return Usage(max(prompt - cached, 0), int(u.get("completion_tokens", 0)), cached, 0)


def _tool_call(tc: dict[str, Any]) -> ToolCall:
    fn = tc.get("function") or {}
    text = fn.get("arguments") or ""
    try:
        parsed = json.loads(text) if text else {}
        malformed = not isinstance(parsed, dict)
    except json.JSONDecodeError:
        parsed, malformed = None, True
    return ToolCall(
        id=str(tc.get("id") or ""),
        name=str(fn.get("name") or ""),
        arguments=parsed if not malformed else None,
        arguments_text=text,
        malformed=malformed,
    )
