"""The Chat Completions adapter: request bodies, message translation, response parsing and
the seed guard, on llama-server-shaped payloads. No network: build_body and parse_turn are
pure, which is why they are tested here rather than through a server."""

from __future__ import annotations

import json

from islands_harness.providers.base import Message, Sampling, StopReason, ToolCall, ToolResult
from islands_harness.providers.openai_compat import (
    LLAMA_RANDOM_SEED,
    OpenAICompatProvider,
    wire_seed,
)
from islands_harness.tools import ToolSpec


async def _noop(args: dict, ctx: object) -> str:
    return ""


FETCH_SPEC = ToolSpec(
    "fetch_document",
    "Fetch.",
    {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
    _noop,
)
SAMPLING = Sampling(
    temperature=1.0, top_p=1.0, max_tokens=512, seed=2**40 + 7, thinking=None, effort=None
)


def _provider(extra: dict | None = None) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        "http://127.0.0.1:8081/v1",
        "gpt-oss-20b",
        api_key_env=None,
        sampling=SAMPLING,
        extra_body=extra,
        allowed_hosts=["127.0.0.1:8081"],
        parallel_tool_calls=True,
    )


def test_wire_seed_is_32_bit_and_never_the_random_sentinel() -> None:
    assert wire_seed(2**40 + 7) == 7
    assert wire_seed(LLAMA_RANDOM_SEED) == LLAMA_RANDOM_SEED - 1
    assert wire_seed((5 << 32) | LLAMA_RANDOM_SEED) == LLAMA_RANDOM_SEED - 1
    assert all(0 <= wire_seed(s) < LLAMA_RANDOM_SEED for s in (0, 1, 2**62, 2**63 - 1))


def test_build_body_translates_every_role_and_keeps_argument_text_verbatim() -> None:
    call = ToolCall(
        id="call_1",
        name="fetch_document",
        arguments={"doc_id": "inv-0001"},
        arguments_text='{"doc_id":"inv-0001"}',
    )
    messages = [
        Message("system", "SYS"),
        Message("user", "Document id: inv-0001"),
        Message(
            "assistant",
            None,
            tool_calls=[call],
            raw_blocks=[{"type": "reasoning_content", "text": "I should fetch it."}],
        ),
        Message(
            "tool",
            tool_results=[
                ToolResult("call_1", "fetch_document", "INVOICE", False, True),
                ToolResult("call_2", "fetch_document", "", False, False),
            ],
        ),
        Message("assistant", "Done."),
    ]
    extra = {
        "samplers": ["top_k", "top_p", "min_p", "temperature"],
        "top_k": 0,
        "min_p": 0.0,
        "cache_prompt": False,
        "reasoning_effort": "medium",
    }
    body = _provider(extra).build_body(messages, [FETCH_SPEC], SAMPLING)
    wire = body["messages"]
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert wire[2]["content"] is None and wire[2]["reasoning_content"] == "I should fetch it."
    assert (
        wire[2]["tool_calls"][0]["function"]["arguments"] == '{"doc_id":"inv-0001"}'
    )  # exactly as written
    assert wire[3] == {"role": "tool", "tool_call_id": "call_1", "content": "INVOICE"}
    assert wire[4]["content"] == ""
    assert body["seed"] == 7 and body["temperature"] == 1.0 and body["max_tokens"] == 512
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is True
    assert "strict" not in json.dumps(body["tools"])
    for key, value in extra.items():  # every pinned sampler reaches the wire
        assert body[key] == value


def test_parse_turn_reads_llama_server_tool_calls_and_reasoning() -> None:
    response = {
        "id": "chatcmpl-x",
        "model": "gpt-oss-20b",
        "system_fingerprint": "b11191-4b1a27fa",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Need the document first.",
                    "tool_calls": [
                        {
                            "id": "abc",
                            "type": "function",
                            "function": {
                                "name": "fetch_document",
                                "arguments": '{"doc_id": "inv-0001"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 812, "completion_tokens": 41},
    }
    turn = OpenAICompatProvider.parse_turn(response, {"model": "gpt-oss-20b"})
    assert turn.stop_reason == StopReason.tool_calls
    assert (
        turn.tool_calls[0].arguments == {"doc_id": "inv-0001"} and not turn.tool_calls[0].malformed
    )
    assert turn.message.content is None
    assert turn.message.raw_blocks == [
        {"type": "reasoning_content", "text": "Need the document first."}
    ]
    assert turn.usage.input_tokens == 812 and turn.usage.output_tokens == 41
    assert turn.identity == {
        "model": "gpt-oss-20b",
        "fingerprint": "b11191-4b1a27fa",
        "response_id": "chatcmpl-x",
    }


def test_parse_turn_flags_malformed_arguments_and_keeps_the_text() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "a",
                            "function": {
                                "name": "submit_record",
                                "arguments": '{"record": {"invoice_number": ',
                            },
                        },
                        {"id": "b", "function": {"name": "submit_record", "arguments": "[1, 2]"}},
                    ],
                },
            }
        ]
    }
    turn = OpenAICompatProvider.parse_turn(response, {})
    assert turn.stop_reason == StopReason.tool_calls  # tool calls win over finish_reason
    bad, not_object = turn.tool_calls
    assert bad.malformed and bad.arguments is None and bad.arguments_text.startswith('{"record"')
    assert not_object.malformed and not_object.arguments is None


def test_parse_turn_maps_finish_reasons_without_tool_calls() -> None:
    def stop(reason: str) -> StopReason:
        return OpenAICompatProvider.parse_turn(
            {"choices": [{"finish_reason": reason, "message": {"content": "x"}}]}, {}
        ).stop_reason

    assert stop("stop") == StopReason.end
    assert stop("length") == StopReason.max_tokens
    assert stop("content_filter") == StopReason.refusal
    assert stop("something_new") == StopReason.other


def test_parse_turn_keeps_length_stop_when_a_tool_call_is_present() -> None:
    """A tool call in a turn cut off at max_tokens may carry partial arguments, so ``length``
    wins over the tool calls and the loop ends the run as limit_output without executing."""

    def turn(reason: str) -> StopReason:
        call = {
            "id": "c0",
            "type": "function",
            "function": {"name": "fetch_document", "arguments": '{"doc_id": "inv-00'},
        }
        message = {"content": "", "tool_calls": [call]}
        body = {"choices": [{"finish_reason": reason, "message": message}]}
        return OpenAICompatProvider.parse_turn(body, {}).stop_reason

    assert turn("length") == StopReason.max_tokens
    assert turn("stop") == StopReason.tool_calls  # servers disagree here; the calls decide
    assert turn("tool_calls") == StopReason.tool_calls
