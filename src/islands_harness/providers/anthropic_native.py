"""Messages API adapter through the official ``anthropic`` SDK (1.8.0, on httpx2), with the
Bedrock, Vertex and Foundry clients of the same SDK.

What is sent (D11, D4), built by the pure ``build_request`` and logged verbatim:

    model, max_tokens, system (the first system Message's text), messages (assistant turns
    replayed from ``raw_blocks`` unchanged, thinking signatures included; tool results as
    one user message of tool_result blocks in call order, ``is_error`` false), tools with
    ``input_schema`` and never ``strict``, tool_choice {type: auto} (never forced),
    thinking {type: adaptive} (omitted when sampling.thinking is None; ``disabled`` is
    refused because Opus 5 then emits tool calls as visible text), output_config {effort},
    and the automatic top-level cache_control {type: ephemeral} when prompt_caching is on.

What is never sent: ``fallbacks`` in any form (a fallback swaps the model under
measurement, D11; this deliberately overrides the SDK guide, which recommends them for
Opus 5), ``temperature``, ``top_p``, ``top_k`` (current models reject them and CAPABILITIES
refuses them), any ``anthropic-beta`` header, any seed (none exists).

Transport: the SDK client is built with ``max_retries=0`` and an httpx2 client whose only
transport is ``AllowlistTransport``, so every retry is the harness's (netguard.with_retry)
and every request is host-checked. A credential is always passed explicitly, because the
SDK's fallback credential chains (profile token exchange, boto3, google-auth) open HTTP
clients of their own that the allowlist cannot see.

Response handling (pure ``parse_turn``): stop_reason end_turn -> end, tool_use ->
tool_calls, max_tokens -> max_tokens, refusal -> refusal, anything else -> other, with
``stop_details`` recorded when present. The response ``model`` and ``id`` and the
``request-id`` header go into identity with the request date, since Anthropic offers no
seed or fingerprint.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import UTC, datetime
from typing import Any, Literal

import anthropic
import httpx2 as httpx

from islands_harness.providers.base import (
    ContextOverflow,
    Message,
    Sampling,
    StopReason,
    ToolCall,
    ToolResult,
    Turn,
    Usage,
    check_capabilities,
)
from islands_harness.providers.netguard import (
    AllowlistTransport,
    HostNotAllowed,
    RetryPolicy,
    TransportError,
    classify_status,
    with_retry,
)
from islands_harness.tools import ToolSpec

Transport = Literal["anthropic", "bedrock", "vertex", "foundry"]

DEFAULT_MODEL = "claude-opus-5"
FIRST_PARTY_URL = "https://api.anthropic.com"

# Default hosts per transport, for doctor's outbound-host listing. Bedrock, Vertex and
# Foundry hosts depend on the region, project or resource and are read from allowed_hosts.
DEFAULT_HOSTS: dict[str, list[str]] = {"anthropic": ["api.anthropic.com"]}

# The environment variable holding the credential when the config names none. Vertex has
# no API key: its credential is an OAuth access token, so the config must name the variable.
_DEFAULT_KEY_ENV: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
    "vertex": None,
    "foundry": "ANTHROPIC_FOUNDRY_API_KEY",
}

# Constructor arguments each transport accepts through ``transport_options``. Anything
# absent falls back to the SDK's own environment variables (AWS_REGION, CLOUD_ML_REGION,
# ANTHROPIC_VERTEX_PROJECT_ID, ANTHROPIC_FOUNDRY_RESOURCE).
_OPTION_KEYS: dict[str, frozenset[str]] = {
    "anthropic": frozenset({"base_url"}),
    "bedrock": frozenset({"aws_region", "base_url"}),
    "vertex": frozenset({"region", "project_id", "base_url"}),
    "foundry": frozenset({"resource", "base_url"}),
}

# Frozen mapping (D11). Anything else, including stop_sequence (never requested), pause_turn
# and model_context_window_exceeded, is ``other``; the raw value stays in identity.
_STOP_MAP: dict[str, StopReason] = {
    "end_turn": StopReason.end,
    "tool_use": StopReason.tool_calls,
    "max_tokens": StopReason.max_tokens,
    "refusal": StopReason.refusal,
}

# Substrings of a 400 error message that mean the request does not fit the context window.
_OVERFLOW_MARKERS = ("prompt is too long", "context window")


# -- client construction ---------------------------------------------------------------


def make_client(
    transport: Transport,
    *,
    allowed_hosts: list[str],
    timeout_s: float,
    api_key_env: str | None = None,
    extra: dict[str, Any] | None = None,
    inner_transport: httpx.AsyncBaseTransport | None = None,
) -> Any:
    """The async SDK client for ``transport``, with ``max_retries=0`` and the allowlisted
    httpx2 client. Every class and argument below was checked against anthropic 1.8.0:

    anthropic:  ``AsyncAnthropic(api_key, base_url)``; base_url defaults to
                api.anthropic.com explicitly, so ANTHROPIC_BASE_URL cannot redirect a run.
    bedrock:    ``AsyncAnthropicBedrockMantle(api_key, aws_region?, base_url?)``, the Messages
                API on Bedrock (bedrock-mantle.{region}.api.aws/anthropic), API-key auth only.
    vertex:     ``AsyncAnthropicVertex(access_token, region?, project_id?, base_url?)``.
    foundry:    ``AsyncAnthropicFoundry(api_key, resource? | base_url?)``.

    ``extra`` holds the transport options listed in ``_OPTION_KEYS``; an unknown key is an
    error, as in the config. ``inner_transport`` replaces the socket transport under the
    allowlist and exists for tests; production passes None. Raises ValueError when the
    credential is missing or the environment injects an ``anthropic-beta`` header.
    """
    if transport not in _OPTION_KEYS:
        raise ValueError(f"unknown anthropic transport {transport!r}")
    options = dict(extra or {})
    unknown = sorted(set(options) - _OPTION_KEYS[transport])
    if unknown:
        raise ValueError(f"transport {transport!r} does not accept options {unknown}")
    credential = _credential(transport, api_key_env)
    common: dict[str, Any] = {
        "max_retries": 0,
        "timeout": httpx.Timeout(timeout_s),
        "http_client": _http_client(allowed_hosts, timeout_s, inner_transport),
    }
    if transport == "anthropic":
        base_url = options.get("base_url", FIRST_PARTY_URL)
        client = anthropic.AsyncAnthropic(api_key=credential, base_url=base_url, **common)
    elif transport == "bedrock":
        client = anthropic.AsyncAnthropicBedrockMantle(api_key=credential, **options, **common)
    elif transport == "vertex":
        client = anthropic.AsyncAnthropicVertex(access_token=credential, **options, **common)
    else:
        client = anthropic.AsyncAnthropicFoundry(api_key=credential, **options, **common)
    _refuse_beta_headers(client)
    return client


def _http_client(
    allowed_hosts: list[str], timeout_s: float, inner: httpx.AsyncBaseTransport | None
) -> httpx.AsyncClient:
    """``DefaultAsyncHttpxClient`` on the allowlist transport. With a custom transport the
    SDK's connection limits no longer reach the socket layer, so they are given to the inner
    transport here; env proxies are not mounted, so nothing can bypass the allowlist."""
    inner = inner or httpx.AsyncHTTPTransport(retries=0, limits=anthropic.DEFAULT_CONNECTION_LIMITS)
    return anthropic.DefaultAsyncHttpxClient(
        transport=AllowlistTransport(allowed_hosts, inner=inner), timeout=httpx.Timeout(timeout_s)
    )


def _credential(transport: str, api_key_env: str | None) -> str:
    """The credential from the environment. Required, so the SDK never falls back to a
    credential chain that makes HTTP requests outside the allowlist."""
    env = api_key_env or _DEFAULT_KEY_ENV[transport]
    if env is None:
        raise ValueError(f"transport {transport!r} needs api_key_env naming its access token")
    value = os.environ.get(env)
    if not value:
        raise ValueError(f"transport {transport!r}: environment variable {env} is not set")
    return value


def _refuse_beta_headers(client: Any) -> None:
    """No beta header is ever sent. ANTHROPIC_CUSTOM_HEADERS could add one to every request
    (a fallback beta, for instance), so a client carrying one is refused at construction."""
    if any(str(name).lower() == "anthropic-beta" for name in client.default_headers):
        raise ValueError("an anthropic-beta header is configured (ANTHROPIC_CUSTOM_HEADERS?)")


# -- provider ----------------------------------------------------------------------------


class AnthropicProvider:
    """ChatProvider over the Messages API. See the module docstring for the frozen rules."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        sampling: Sampling,
        prompt_caching: bool,
        allowed_hosts: list[str],
        transport: Transport | None = "anthropic",
        api_key_env: str | None = None,
        transport_options: dict[str, Any] | None = None,
        timeout_s: float = 600.0,
        parallel_tool_calls: bool = True,
        retry_policy: RetryPolicy = RetryPolicy(),
        log: Any = None,
        client: Any = None,
    ) -> None:
        _check_sampling(sampling)
        self.model = model
        self.transport: Transport = transport or "anthropic"
        self.sampling = sampling
        self.prompt_caching = prompt_caching
        self.parallel_tool_calls = parallel_tool_calls
        self.retry_policy = retry_policy
        self.log = log or (lambda event: None)
        self.client = client or make_client(
            self.transport,
            allowed_hosts=allowed_hosts,
            timeout_s=timeout_s,
            api_key_env=api_key_env,
            extra=transport_options,
        )

    # -- translation -------------------------------------------------------------------

    def build_request(
        self, messages: list[Message], tools: list[ToolSpec], sampling: Sampling
    ) -> dict[str, Any]:
        """The kwargs for ``client.messages.create``, which are also the JSON body the
        first-party API receives (Vertex moves ``model`` into the URL). Pure.

        tool_choice is exactly {"type": "auto"}; ``disable_parallel_tool_use`` is added only
        when the config disallows parallel calls. No sampling field beyond max_tokens,
        thinking and effort can appear, and ``fallbacks`` never does.
        """
        _check_sampling(sampling)
        system, rest = _split_system(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": sampling.max_tokens,
            "messages": [_to_wire(m) for m in rest],
        }
        if system is not None:
            request["system"] = system
        if tools:
            request["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
            choice: dict[str, Any] = {"type": "auto"}
            if not self.parallel_tool_calls:
                choice["disable_parallel_tool_use"] = True
            request["tool_choice"] = choice
        if sampling.thinking == "adaptive":
            request["thinking"] = {"type": "adaptive"}
        if sampling.effort is not None:
            request["output_config"] = {"effort": sampling.effort}
        if self.prompt_caching:
            request["cache_control"] = {"type": "ephemeral"}
        return request

    @staticmethod
    def parse_turn(response: Any, request: dict[str, Any]) -> Turn:
        """Response body (a dict, or an SDK Message) -> Turn. Pure.

        Every content block is kept unchanged in ``message.raw_blocks`` for exact replay;
        text blocks join with newlines; each tool_use block becomes a ToolCall whose
        ``arguments_text`` is the canonical JSON of its input (sorted keys), and an input
        that is not an object is flagged ``malformed``.
        """
        body = _response_body(response)
        blocks: list[dict[str, Any]] = list(body.get("content") or [])
        text = "\n".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
        calls = [_tool_call(b) for b in blocks if b.get("type") == "tool_use"]
        raw_stop = body.get("stop_reason")
        identity: dict[str, Any] = {
            "model": body.get("model"),
            "response_id": body.get("id"),
            "stop_reason_raw": raw_stop,
        }
        if body.get("stop_details") is not None:
            identity["stop_details"] = body["stop_details"]
        message = Message(
            role="assistant",
            content=text or None,
            tool_calls=calls or None,
            raw_blocks=copy.deepcopy(blocks),
        )
        return Turn(
            text=text,
            tool_calls=calls,
            usage=_usage(body.get("usage")),
            stop_reason=_STOP_MAP.get(raw_stop or "", StopReason.other),
            identity=identity,
            message=message,
            raw_request=request,
            raw_response=body,
        )

    # -- transport ---------------------------------------------------------------------

    async def _create(self, request: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """One SDK call, returning the response body exactly as the API sent it and the
        ``request-id`` header. SDK errors map to netguard's classification:

        - 400 whose message says the prompt does not fit: ContextOverflow.
        - Any other status: TransportError, retryable per ``classify_status`` (408, 409,
          429, 5xx including 529), exactly as the OpenAI-compatible adapter classifies.
        - Connection errors and timeouts: retryable TransportError, except a refusal by
          the allowlist, which the SDK wraps as a connection error and which is re-raised
          as HostNotAllowed so it is never retried.
        """
        try:
            raw = await self.client.messages.with_raw_response.create(**request)
            body = await raw.json()
        except anthropic.APIStatusError as exc:
            message = _error_message(exc)
            if exc.status_code == 400 and _is_overflow(message):
                raise ContextOverflow(message[:500]) from exc
            raise TransportError(
                f"HTTP {exc.status_code}: {message[:300]}",
                retryable=classify_status(exc.status_code),
                status=exc.status_code,
            ) from exc
        except anthropic.APIConnectionError as exc:
            if isinstance(exc.__cause__, HostNotAllowed):
                raise exc.__cause__ from None
            raise TransportError(f"{type(exc).__name__}: {exc}", retryable=True) from exc
        if not isinstance(body, dict):
            raise TransportError("response body is not a JSON object", retryable=False)
        return body, raw.request_id

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec], sampling: Sampling
    ) -> Turn:
        request = self.build_request(messages, tools, sampling)
        retries = {"n": 0}

        def log(event: dict[str, Any]) -> None:
            retries["n"] += 1
            self.log(event)

        sent_at = datetime.now(UTC).isoformat()
        body, request_id = await with_retry(lambda: self._create(request), self.retry_policy, log)
        turn = self.parse_turn(body, request)
        identity = dict(turn.identity)
        identity.update(
            {
                "request_date": sent_at,
                "request_id": request_id,
                "transport": self.transport,
                "effort": sampling.effort,
                "thinking": sampling.thinking,
            }
        )
        return Turn(
            text=turn.text,
            tool_calls=turn.tool_calls,
            usage=turn.usage,
            stop_reason=turn.stop_reason,
            identity=identity,
            message=turn.message,
            raw_request=request,
            raw_response=body,
            transport_retries=retries["n"],
        )

    def identity(self) -> dict[str, Any]:
        return {
            "provider": "anthropic",
            "transport": self.transport,
            "model": self.model,
            "thinking": self.sampling.thinking,
            "effort": self.sampling.effort,
            "prompt_caching": self.prompt_caching,
            "fallbacks": "none",
            "sdk": f"anthropic {anthropic.__version__}",
        }

    async def aclose(self) -> None:
        await self.client.close()


# -- helpers -----------------------------------------------------------------------------


def _check_sampling(sampling: Sampling) -> None:
    """CAPABILITIES refuses temperature, top_p and seed; thinking=disabled is refused (D11)."""
    check_capabilities("anthropic", sampling)
    if sampling.thinking == "disabled":
        raise ValueError("thinking=disabled is refused (D11): use adaptive with a lower effort")


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """The leading system Message's text and the rest. The loop sends exactly one system
    message, first; a system message anywhere else is an error, not a silent reorder."""
    if messages and messages[0].role == "system":
        system, rest = messages[0].content or "", messages[1:]
    else:
        system, rest = None, messages
    if any(m.role == "system" for m in rest):
        raise ValueError("a system message is only accepted as the first message")
    return system, rest


def _to_wire(m: Message) -> dict[str, Any]:
    """Provider-neutral Message -> Messages API message.

    Assistant turns are replayed from ``raw_blocks`` verbatim (a copy, so the logged
    request never aliases the transcript). A tool Message becomes one user message of
    tool_result blocks in call order, ``is_error`` false: faults and model errors are
    delivered as ordinary results by design.
    """
    if m.role == "user":
        return {"role": "user", "content": m.content or ""}
    if m.role == "assistant":
        blocks = m.raw_blocks if m.raw_blocks is not None else _blocks_from_fields(m)
        return {"role": "assistant", "content": copy.deepcopy(blocks)}
    if m.role == "tool":
        return {"role": "user", "content": [_tool_result(r) for r in (m.tool_results or [])]}
    raise ValueError(f"unexpected message role {m.role!r}")


def _tool_result(r: ToolResult) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": r.call_id,
        "content": r.content,
        "is_error": False,
    }


def _blocks_from_fields(m: Message) -> list[dict[str, Any]]:
    """Blocks for an assistant Message this adapter did not produce (no ``raw_blocks``).
    A malformed call cannot be rebuilt faithfully, so it is refused."""
    blocks: list[dict[str, Any]] = []
    if m.content:
        blocks.append({"type": "text", "text": m.content})
    for c in m.tool_calls or []:
        if c.malformed or c.arguments is None:
            raise ValueError(f"tool call {c.id!r} is malformed and has no raw block to replay")
        blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
    return blocks


def _response_body(response: Any) -> dict[str, Any]:
    """The response as the API's JSON object. SDK models dump with the API's own keys and
    only the fields the API set, which is what the SDK itself sends back on replay."""
    if isinstance(response, dict):
        return response
    return response.to_dict(mode="json", warnings=False)


def _tool_call(block: dict[str, Any]) -> ToolCall:
    value = block.get("input")
    malformed = not isinstance(value, dict)
    return ToolCall(
        id=str(block.get("id") or ""),
        name=str(block.get("name") or ""),
        arguments=None if malformed else value,
        arguments_text=json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        malformed=malformed,
    )


def _usage(u: dict[str, Any] | None) -> Usage:
    """input_tokens is the API's uncached remainder; cache reads and writes are separate."""
    u = u or {}
    return Usage(
        input_tokens=int(u.get("input_tokens") or 0),
        output_tokens=int(u.get("output_tokens") or 0),
        cache_read_tokens=int(u.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
    )


def _error_message(exc: anthropic.APIStatusError) -> str:
    """The API's error message from the body, or the SDK's message when the body has none."""
    body = exc.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message")
        if isinstance(message, str):
            return message
    return str(exc.message)


def _is_overflow(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)
