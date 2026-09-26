"""The Messages API adapter: request building, response parsing, error mapping and the
allowlist, with no network and no API key.

Pure functions are tested on payloads shaped like the documented Messages API response.
The client path runs through the real anthropic SDK client with an httpx2 MockTransport
under the AllowlistTransport, so what is asserted is what the SDK actually puts on the wire.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from islands_harness.providers.anthropic_native import (
    DEFAULT_MODEL,
    AnthropicProvider,
    make_client,
)
from islands_harness.providers.base import (
    ContextOverflow,
    Message,
    Sampling,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)
from islands_harness.providers.netguard import (
    HostNotAllowed,
    RetryPolicy,
    TransportError,
    TransportExhausted,
)
from islands_harness.tools import ToolSpec

HOST = "api.anthropic.com"
FAST = RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)
SAMPLING = Sampling(
    temperature=None, top_p=None, max_tokens=4096, seed=None, thinking="adaptive", effort="medium"
)
FETCH_SCHEMA = {
    "type": "object",
    "properties": {"doc_id": {"type": "string"}},
    "required": ["doc_id"],
}
SUBMIT_SCHEMA = {"type": "object", "properties": {"record": {"type": "object"}}}

# Environment variables the SDK reads that could change where or how a request goes.
SDK_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
    "CLOUD_ML_REGION",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "ANTHROPIC_FOUNDRY_BASE_URL",
)


async def _noop(args: dict, ctx: object) -> str:
    return ""


TOOLS = [
    ToolSpec("fetch_document", "Fetch a document.", FETCH_SCHEMA, _noop),
    ToolSpec("submit_record", "Submit the record.", SUBMIT_SCHEMA, _noop),
]

THINKING = {"type": "thinking", "thinking": "", "signature": "EqQBCkgIBxABGAIiQGsig=="}
TEXT = {"type": "text", "text": "Fetching the document."}
TOOL_USE = {
    "type": "tool_use",
    "id": "toolu_01A",
    "name": "fetch_document",
    "input": {"doc_id": "inv-0001"},
    "caller": {"type": "direct"},
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real credential or redirect from the developer's shell reaches a test."""
    for name in SDK_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-real-key")


def _body(
    content: list[dict[str, Any]],
    stop_reason: str | None = "end_turn",
    *,
    usage: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """A response body in the documented Messages API shape."""
    return {
        "id": "msg_01XYZ",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage
        or {
            "input_tokens": 1200,
            "output_tokens": 85,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 300,
        },
        **fields,
    }


def _error(status: int, message: str, kind: str = "invalid_request_error") -> httpx.Response:
    return httpx.Response(
        status,
        json={"type": "error", "error": {"type": kind, "message": message}, "request_id": "req_e"},
    )


Step = httpx.Response | Exception | dict[str, Any]


class FakeAPI:
    """A scripted Messages API behind MockTransport. Each request pops one step: a response
    body (sent as 200), an httpx2.Response, or an exception raised as the transport would."""

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        return httpx.Response(200, json=step, headers={"request-id": "req_011abc"})

    def sent(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _provider(
    api: FakeAPI | None = None,
    *,
    caching: bool = True,
    parallel: bool = True,
    sampling: Sampling = SAMPLING,
    hosts: tuple[str, ...] = (HOST,),
    options: dict[str, Any] | None = None,
    log: Callable[[dict[str, Any]], None] | None = None,
) -> AnthropicProvider:
    client = make_client(
        "anthropic",
        allowed_hosts=list(hosts),
        timeout_s=30.0,
        extra=options,
        inner_transport=httpx.MockTransport(api or FakeAPI()),
    )
    return AnthropicProvider(
        DEFAULT_MODEL,
        sampling=sampling,
        prompt_caching=caching,
        allowed_hosts=list(hosts),
        parallel_tool_calls=parallel,
        retry_policy=FAST,
        log=log,
        client=client,
    )


def _conversation() -> list[Message]:
    call = ToolCall("toolu_01A", "fetch_document", {"doc_id": "inv-0001"}, '{"doc_id":"inv-0001"}')
    return [
        Message("system", "SYS"),
        Message("user", "Document id: inv-0001"),
        Message(
            "assistant",
            "Fetching the document.",
            tool_calls=[call],
            raw_blocks=[THINKING, TEXT, TOOL_USE],
        ),
        Message(
            "tool",
            tool_results=[
                ToolResult("toolu_01A", "fetch_document", "INVOICE TEXT", False, True),
                ToolResult("toolu_01B", "search_web", "", False, False),
                ToolResult("toolu_01C", "nope", "Unknown tool.", True, False, model_error=True),
            ],
        ),
    ]


# -- request building ----------------------------------------------------------------------


def test_request_translates_every_role_and_replays_assistant_blocks_verbatim() -> None:
    messages = _conversation()
    request = _provider().build_request(messages, TOOLS, SAMPLING)

    assert request["system"] == "SYS"
    assert request["messages"] == [
        {"role": "user", "content": "Document id: inv-0001"},
        {"role": "assistant", "content": [THINKING, TEXT, TOOL_USE]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A",
                    "content": "INVOICE TEXT",
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01B",
                    "content": "",
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01C",
                    "content": "Unknown tool.",
                    "is_error": False,
                },
            ],
        },
    ]
    # The logged request is a copy: mutating it cannot reach the transcript.
    request["messages"][1]["content"][0]["signature"] = "changed"
    assert messages[2].raw_blocks[0]["signature"] == THINKING["signature"]


def test_request_carries_exactly_the_snapshot_configuration() -> None:
    request = _provider().build_request(_conversation(), TOOLS, SAMPLING)
    request.pop("messages")
    assert request == {
        "model": "claude-opus-5",
        "max_tokens": 4096,
        "system": "SYS",
        "tools": [
            {
                "name": "fetch_document",
                "description": "Fetch a document.",
                "input_schema": FETCH_SCHEMA,
            },
            {
                "name": "submit_record",
                "description": "Submit the record.",
                "input_schema": SUBMIT_SCHEMA,
            },
        ],
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
        "cache_control": {"type": "ephemeral"},
    }


@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_request_never_carries_fallbacks_strict_or_sampling_fields(
    caching: bool, parallel: bool
) -> None:
    request = _provider(caching=caching, parallel=parallel).build_request(
        _conversation(), TOOLS, SAMPLING
    )
    for key in ("fallbacks", "temperature", "top_p", "top_k", "seed", "stop_sequences", "betas"):
        assert key not in request
    assert "fallback" not in json.dumps(request)
    assert all("strict" not in tool for tool in request["tools"])
    assert request["tool_choice"]["type"] == "auto"


def test_disallowed_parallel_calls_is_the_only_tool_choice_variation() -> None:
    allowed = _provider(parallel=True).build_request(_conversation(), TOOLS, SAMPLING)
    disallowed = _provider(parallel=False).build_request(_conversation(), TOOLS, SAMPLING)
    assert allowed["tool_choice"] == {"type": "auto"}
    assert disallowed["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_caching_off_and_unset_thinking_or_effort_are_omitted() -> None:
    sampling = Sampling(None, None, 2048, None, thinking=None, effort=None)
    request = _provider(caching=False, sampling=sampling).build_request(
        _conversation(), [], sampling
    )
    for key in ("cache_control", "thinking", "output_config", "tools", "tool_choice"):
        assert key not in request
    assert request["max_tokens"] == 2048


@pytest.mark.parametrize(
    "sampling",
    [
        Sampling(0.7, None, 4096, None, "adaptive", "medium"),
        Sampling(None, 1.0, 4096, None, "adaptive", "medium"),
        Sampling(None, None, 4096, 7, "adaptive", "medium"),
        Sampling(None, None, 4096, None, "disabled", "medium"),
    ],
    ids=["temperature", "top_p", "seed", "thinking_disabled"],
)
def test_refused_sampling_fields_fail_at_construction_and_at_build(sampling: Sampling) -> None:
    with pytest.raises(ValueError):
        AnthropicProvider(
            sampling=sampling, prompt_caching=True, allowed_hosts=[HOST], client=object()
        )
    with pytest.raises(ValueError):
        _provider().build_request(_conversation(), TOOLS, sampling)


def test_assistant_without_raw_blocks_is_rebuilt_and_a_malformed_call_is_refused() -> None:
    call = ToolCall("toolu_9", "fetch_document", {"doc_id": "x"}, '{"doc_id":"x"}')
    rebuilt = _provider().build_request(
        [Message("user", "u"), Message("assistant", "t", tool_calls=[call])], TOOLS, SAMPLING
    )
    assert rebuilt["messages"][1]["content"] == [
        {"type": "text", "text": "t"},
        {"type": "tool_use", "id": "toolu_9", "name": "fetch_document", "input": {"doc_id": "x"}},
    ]
    bad = ToolCall("toolu_9", "fetch_document", None, "[1]", malformed=True)
    with pytest.raises(ValueError):
        _provider().build_request(
            [Message("user", "u"), Message("assistant", None, tool_calls=[bad])], TOOLS, SAMPLING
        )


def test_a_system_message_is_only_accepted_first() -> None:
    with pytest.raises(ValueError):
        _provider().build_request(
            [Message("user", "u"), Message("system", "late")], TOOLS, SAMPLING
        )
    request = _provider().build_request([Message("user", "u")], TOOLS, SAMPLING)
    assert "system" not in request


# -- response parsing ----------------------------------------------------------------------


def test_parse_turn_keeps_every_block_and_maps_usage() -> None:
    second = {
        "type": "tool_use",
        "id": "toolu_01B",
        "name": "submit_record",
        "input": {"z": 1, "a": {"y": 2, "b": 3}},
    }
    body = _body([THINKING, TEXT, TOOL_USE, {"type": "text", "text": "More."}, second], "tool_use")
    turn = AnthropicProvider.parse_turn(body, {"model": "x"})

    assert turn.stop_reason == StopReason.tool_calls
    assert turn.text == "Fetching the document.\nMore."
    assert turn.message.content == turn.text
    assert turn.message.raw_blocks == body["content"]
    assert turn.message.raw_blocks is not body["content"]
    assert turn.message.raw_blocks[0]["signature"] == THINKING["signature"]
    assert [(c.id, c.name, c.arguments, c.malformed) for c in turn.tool_calls] == [
        ("toolu_01A", "fetch_document", {"doc_id": "inv-0001"}, False),
        ("toolu_01B", "submit_record", {"z": 1, "a": {"y": 2, "b": 3}}, False),
    ]
    assert turn.tool_calls[1].arguments_text == '{"a":{"b":3,"y":2},"z":1}'
    assert turn.message.tool_calls == turn.tool_calls
    assert turn.usage == Usage(1200, 85, 900, 300)
    assert turn.identity == {
        "model": "claude-opus-5",
        "response_id": "msg_01XYZ",
        "stop_reason_raw": "tool_use",
    }
    assert turn.raw_response is body
    assert turn.raw_request == {"model": "x"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", StopReason.end),
        ("tool_use", StopReason.tool_calls),
        ("max_tokens", StopReason.max_tokens),
        ("refusal", StopReason.refusal),
        ("stop_sequence", StopReason.other),
        ("pause_turn", StopReason.other),
        ("model_context_window_exceeded", StopReason.other),
        ("something_new", StopReason.other),
        (None, StopReason.other),
    ],
)
def test_stop_reason_mapping(raw: str | None, expected: StopReason) -> None:
    turn = AnthropicProvider.parse_turn(_body([TEXT], raw), {})
    assert turn.stop_reason == expected
    assert turn.identity["stop_reason_raw"] == raw


def test_refusal_records_stop_details_and_tolerates_empty_content() -> None:
    details = {"type": "refusal", "category": "cyber", "explanation": None}
    body = _body([], "refusal", usage={"input_tokens": 0, "output_tokens": 0}, stop_details=details)
    turn = AnthropicProvider.parse_turn(body, {})
    assert turn.stop_reason == StopReason.refusal
    assert turn.identity["stop_details"] == details
    assert turn.text == "" and turn.tool_calls == [] and turn.message.content is None
    assert turn.message.raw_blocks == []
    assert turn.usage == Usage(0, 0, 0, 0)
    assert "stop_details" not in AnthropicProvider.parse_turn(_body([TEXT]), {}).identity


@pytest.mark.parametrize("value", ["not an object", [1, 2], None, 3])
def test_non_object_tool_input_is_malformed_with_canonical_text(value: Any) -> None:
    block = {"type": "tool_use", "id": "toolu_X", "name": "fetch_document", "input": value}
    (call,) = AnthropicProvider.parse_turn(_body([block], "tool_use"), {}).tool_calls
    assert call.malformed is True
    assert call.arguments is None
    assert call.arguments_text == json.dumps(value, separators=(",", ":"))


def test_a_parsed_turn_replays_its_blocks_unchanged_on_the_next_request() -> None:
    body = _body([THINKING, TEXT, TOOL_USE], "tool_use")
    original = copy.deepcopy(body["content"])
    turn = AnthropicProvider.parse_turn(body, {})
    messages = [Message("system", "SYS"), Message("user", "u"), turn.message]
    replayed = _provider().build_request(messages, TOOLS, SAMPLING)["messages"][1]
    assert replayed == {"role": "assistant", "content": original}


# -- the SDK client path -------------------------------------------------------------------


async def test_complete_sends_exactly_the_logged_request_through_the_sdk() -> None:
    api = FakeAPI(_body([THINKING, TEXT, TOOL_USE], "tool_use"))
    provider = _provider(api)
    turn = await provider.complete(_conversation(), TOOLS, SAMPLING)

    (request,) = api.requests
    assert request.url == httpx.URL("https://api.anthropic.com/v1/messages")
    assert json.loads(request.content) == turn.raw_request
    assert request.headers["x-api-key"] == "test-key-not-a-real-key"
    assert request.headers["x-stainless-retry-count"] == "0"
    assert turn.raw_response == _body([THINKING, TEXT, TOOL_USE], "tool_use")
    assert turn.message.raw_blocks == [THINKING, TEXT, TOOL_USE]
    assert turn.stop_reason == StopReason.tool_calls
    assert turn.transport_retries == 0
    assert turn.identity["model"] == "claude-opus-5"
    assert turn.identity["response_id"] == "msg_01XYZ"
    assert turn.identity["request_id"] == "req_011abc"
    assert turn.identity["request_date"]
    assert (turn.identity["transport"], turn.identity["effort"], turn.identity["thinking"]) == (
        "anthropic",
        "medium",
        "adaptive",
    )
    assert provider.client.max_retries == 0
    await provider.aclose()


@pytest.mark.parametrize("caching", [True, False])
async def test_no_fallbacks_key_and_no_beta_header_on_the_wire(caching: bool) -> None:
    api = FakeAPI(_body([TEXT]))
    provider = _provider(api, caching=caching)
    await provider.complete(_conversation(), TOOLS, SAMPLING)

    sent = api.sent()
    assert "fallbacks" not in sent
    assert "fallback" not in api.requests[0].content.decode("utf-8")
    assert "anthropic-beta" not in api.requests[0].headers
    assert not any("fallback" in v.lower() for v in api.requests[0].headers.values())
    assert ("cache_control" in sent) is caching
    assert provider.identity()["fallbacks"] == "none"


async def test_parse_turn_accepts_the_sdk_message_model() -> None:
    body = _body([THINKING, TEXT, TOOL_USE], "tool_use")
    client = make_client(
        "anthropic",
        allowed_hosts=[HOST],
        timeout_s=30.0,
        inner_transport=httpx.MockTransport(FakeAPI(body)),
    )
    message = await client.messages.create(
        model=DEFAULT_MODEL, max_tokens=16, messages=[{"role": "user", "content": "u"}]
    )
    assert isinstance(message, anthropic.types.Message)
    turn = AnthropicProvider.parse_turn(message, {})
    assert turn.message.raw_blocks == body["content"]
    assert turn.usage == Usage(1200, 85, 900, 300)
    await client.close()


# -- error mapping -------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503, 529])
async def test_retryable_statuses_are_retried_by_the_harness_not_the_sdk(status: int) -> None:
    events: list[dict[str, Any]] = []
    api = FakeAPI(_error(status, "try again", "rate_limit_error"), _body([TEXT]))
    turn = await _provider(api, log=events.append).complete(_conversation(), TOOLS, SAMPLING)
    assert turn.text == TEXT["text"]
    assert turn.transport_retries == 1
    assert len(api.requests) == 2
    assert [r.headers["x-stainless-retry-count"] for r in api.requests] == ["0", "0"]
    assert events[0]["event"] == "transport_retry" and events[0]["status"] == status


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
async def test_client_errors_are_not_retried(status: int) -> None:
    api = FakeAPI(_error(status, "messages: roles must alternate"))
    with pytest.raises(TransportError) as info:
        await _provider(api).complete(_conversation(), TOOLS, SAMPLING)
    assert info.value.retryable is False
    assert info.value.status == status
    assert len(api.requests) == 1


@pytest.mark.parametrize(
    "message",
    [
        "prompt is too long: 1000123 tokens > 1000000 maximum",
        "input length and `max_tokens` exceed context window",
    ],
)
async def test_a_prompt_too_long_400_is_context_overflow(message: str) -> None:
    api = FakeAPI(_error(400, message))
    with pytest.raises(ContextOverflow):
        await _provider(api).complete(_conversation(), TOOLS, SAMPLING)
    assert len(api.requests) == 1


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("connection refused"), httpx.ReadTimeout("read timed out")],
    ids=["connect", "timeout"],
)
async def test_connection_errors_and_timeouts_are_retryable(error: Exception) -> None:
    api = FakeAPI(error, _body([TEXT]))
    turn = await _provider(api).complete(_conversation(), TOOLS, SAMPLING)
    assert turn.transport_retries == 1
    assert len(api.requests) == 2


async def test_exhausted_retries_raise_transport_exhausted() -> None:
    api = FakeAPI(
        *[_error(529, "overloaded", "overloaded_error") for _ in range(FAST.max_attempts)]
    )
    with pytest.raises(TransportExhausted):
        await _provider(api).complete(_conversation(), TOOLS, SAMPLING)
    assert len(api.requests) == FAST.max_attempts


async def test_allowlist_refuses_a_non_allowed_host_through_the_sdk_client() -> None:
    events: list[dict[str, Any]] = []
    api = FakeAPI(_body([TEXT]))
    provider = _provider(api, options={"base_url": "https://evil.example.com"}, log=events.append)
    with pytest.raises(HostNotAllowed):
        await provider.complete(_conversation(), TOOLS, SAMPLING)
    assert api.requests == []
    assert events == []


# -- client construction -------------------------------------------------------------------


def test_make_client_requires_an_explicit_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_client("anthropic", allowed_hosts=[HOST], timeout_s=30.0)
    with pytest.raises(ValueError, match="api_key_env"):
        make_client("vertex", allowed_hosts=[HOST], timeout_s=30.0)


def test_make_client_refuses_an_injected_beta_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS", "anthropic-beta: server-side-fallback-2026-07-01"
    )
    with pytest.raises(ValueError, match="anthropic-beta"):
        make_client("anthropic", allowed_hosts=[HOST], timeout_s=30.0)


def test_make_client_refuses_unknown_options_and_transports() -> None:
    with pytest.raises(ValueError, match="fallbacks"):
        make_client(
            "anthropic", allowed_hosts=[HOST], timeout_s=30.0, extra={"fallbacks": "default"}
        )
    with pytest.raises(ValueError):
        make_client("openai", allowed_hosts=[HOST], timeout_s=30.0)  # type: ignore[arg-type]


async def test_base_url_environment_cannot_redirect_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example.com")
    api = FakeAPI(_body([TEXT]))
    await _provider(api).complete(_conversation(), TOOLS, SAMPLING)
    assert api.requests[0].url.host == HOST


@pytest.mark.parametrize(
    ("transport", "env", "options", "url", "client_cls"),
    [
        (
            "bedrock",
            {"AWS_BEARER_TOKEN_BEDROCK": "test-bedrock-key"},
            {"aws_region": "us-east-1"},
            "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
            anthropic.AsyncAnthropicBedrockMantle,
        ),
        (
            "vertex",
            {"TEST_VERTEX_TOKEN": "test-vertex-token"},
            {"region": "us-east5", "project_id": "proj"},
            "https://us-east5-aiplatform.googleapis.com/v1/projects/proj/locations/us-east5"
            "/publishers/anthropic/models/claude-opus-5:rawPredict",
            anthropic.AsyncAnthropicVertex,
        ),
        (
            "foundry",
            {"ANTHROPIC_FOUNDRY_API_KEY": "test-foundry-key"},
            {"resource": "islands"},
            "https://islands.services.ai.azure.com/anthropic/v1/messages",
            anthropic.AsyncAnthropicFoundry,
        ),
    ],
)
async def test_platform_clients_go_through_the_allowlist_without_retries(
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
    env: dict[str, str],
    options: dict[str, str],
    url: str,
    client_cls: type,
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    key_env = "TEST_VERTEX_TOKEN" if transport == "vertex" else None
    host = httpx.URL(url).host
    api = FakeAPI(_body([TEXT]))
    client = make_client(
        transport,  # type: ignore[arg-type]
        allowed_hosts=[host],
        timeout_s=30.0,
        api_key_env=key_env,
        extra=options,
        inner_transport=httpx.MockTransport(api),
    )
    assert isinstance(client, client_cls)
    assert client.max_retries == 0
    provider = AnthropicProvider(
        sampling=SAMPLING,
        prompt_caching=True,
        allowed_hosts=[host],
        transport=transport,  # type: ignore[arg-type]
        retry_policy=FAST,
        client=client,
    )
    turn = await provider.complete(_conversation(), TOOLS, SAMPLING)
    assert str(api.requests[0].url) == url
    assert "fallbacks" not in api.sent()
    assert "anthropic-beta" not in api.requests[0].headers
    assert turn.identity["transport"] == transport

    blocked = make_client(
        transport,  # type: ignore[arg-type]
        allowed_hosts=["api.anthropic.com"],
        timeout_s=30.0,
        api_key_env=key_env,
        extra=options,
        inner_transport=httpx.MockTransport(api),
    )
    with pytest.raises(anthropic.APIConnectionError) as info:
        await blocked.messages.create(
            model=DEFAULT_MODEL, max_tokens=16, messages=[{"role": "user", "content": "u"}]
        )
    assert isinstance(info.value.__cause__, HostNotAllowed)
    assert len(api.requests) == 1


def test_static_identity() -> None:
    assert _provider().identity() == {
        "provider": "anthropic",
        "transport": "anthropic",
        "model": "claude-opus-5",
        "thinking": "adaptive",
        "effort": "medium",
        "prompt_caching": True,
        "fallbacks": "none",
        "sdk": f"anthropic {anthropic.__version__}",
    }
