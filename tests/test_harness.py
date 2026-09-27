"""Unit and property tests for the run path.

Tests whose code is still a TODO are marked ``xfail(raises=NotImplementedError, strict=True)``:
they fail visibly today, and the moment the logic lands they must pass or the strict marker
turns into a failure that says "remove me". Expected values for the rng tests were computed
independently (Node's crypto over the same canonical encoding) so the Python implementation
is checked against something other than itself.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import httpx2 as httpx
import pytest
from pydantic import ValidationError

from islands_harness import loop as loop_module
from islands_harness import rng
from islands_harness.config import (
    AgentConfig,
    FaultConfig,
    GarbleConfig,
    LockMismatch,
    canonical_bytes,
    load_config,
    load_models,
    load_prereg,
    sha256_file,
)
from islands_harness.faults import FaultInjector, FaultKind
from islands_harness.loop import Outcome, Prompts, run_one
from islands_harness.providers.base import (
    FakeProvider,
    Sampling,
    ScriptedTurn,
    StopReason,
    ToolCall,
)
from islands_harness.providers.netguard import (
    AllowlistTransport,
    HostNotAllowed,
    RetryPolicy,
    TransportError,
    TransportExhausted,
    with_retry,
)
from islands_harness.runner import Storage
from islands_harness.specs import Cell, RunSpec, expand_cells, run_specs
from islands_harness.tools import (
    MALFORMED_ARGUMENTS_TEXT,
    Registry,
    RunContext,
    ToolSpec,
    validate_arguments,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "snapshot-1.yaml"
PREREG = REPO / "configs" / "preregistration" / "r4.yaml"

TODO = pytest.mark.xfail  # alias used with raises=NotImplementedError, strict=True


# -- rng --------------------------------------------------------------------------------


def test_rng_fixed_values_match_independent_implementation() -> None:
    # The scope strings are fixed test vectors, independent of the model ids in the config.
    assert rng.unit("sample", 20261006, "qwen3-4b-local", 0) == pytest.approx(
        0.50855578268573076, abs=1e-14
    )
    assert rng.unit("sample", 20261006, "qwen3-4b-local", 1) == pytest.approx(
        0.72182307316930916, abs=1e-14
    )
    assert rng.u32("sample", 20261006, "qwen3-4b-local", 0) == 2184230454
    assert rng.u32("sample", 20261006, "qwen3-4b-local", 1) == 3100206492
    assert rng.seed64("sample", 20261006, "qwen3-4b-local", 0) == 4690599185204363955
    assert rng.seed64("sample", 20261006, "qwen3-4b-local", 1) == 6657642748626384917
    assert rng.unit("fault", 20261006, "qwen3-4b-local", "A", "inv-0007", 0, 3) == pytest.approx(
        0.88251296997453232, abs=1e-14
    )
    assert rng.unit("kind", 20261006, "qwen3-4b-local", "A", "inv-0007", 0, 3) == pytest.approx(
        0.36284975376678452, abs=1e-14
    )
    assert rng.unit("perm", 20261006, "qwen3-4b-local", 5) == pytest.approx(
        0.57114532574801502, abs=1e-14
    )


def test_rng_is_stateless_and_in_range() -> None:
    values = [rng.unit("t", i) for i in range(2000)]
    assert values == [rng.unit("t", i) for i in range(2000)]
    assert all(0.0 <= v < 1.0 for v in values)
    assert rng.seed64("t", 1) < 2**63
    assert rng.unit("a", 1) != rng.unit("b", 1)


def test_choice_weighted_is_monotone_in_u() -> None:
    items = ["timeout", "garbled_payload", "error_response", "empty_result"]
    picks = [rng.choice_weighted(u / 100, items, [1, 1, 1, 1]) for u in range(100)]
    assert (
        picks
        == ["timeout"] * 25
        + ["garbled_payload"] * 25
        + ["error_response"] * 25
        + ["empty_result"] * 25
    )


# -- config and hashing -------------------------------------------------------------------


def test_example_config_and_prereg_load() -> None:
    cfg = load_config(CONFIG)
    assert cfg.snapshot.id == "s1"
    hosted = {m.id: m for m in load_models(REPO / "configs" / "hosted-models.yaml")}
    assert hosted["opus5-hosted"].fallbacks == "none"
    assert all(m.provider == "openai_compat" for m in cfg.models)  # snapshot 1 is local only
    assert cfg.model_by_id("gpt-oss-20b-local").sampling.seed == "per_run"
    prereg = load_prereg(PREREG)
    assert prereg.version == "r4"
    assert prereg.p1.ladder == ["mle_cluster", "firth_bootstrap", "not_estimable"]


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8") + "\nunexpected_knob: 1\n"
    bad = tmp_path / "bad.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="unexpected_knob"):
        load_config(bad)


def test_sha256_file_normalizes_crlf(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes(b"a\nb\n")
    crlf.write_bytes(b"a\r\nb\r\n")
    expected = "911169ddaaf146aff539f58c26c489af3b892dff0fe283c1c264c65ae5aa59a2"
    assert sha256_file(lf) == expected
    assert sha256_file(crlf) == expected
    assert sha256_file(crlf, normalize_newlines=False) != expected


def test_canonical_bytes_sorts_keys_and_posixifies_paths() -> None:
    assert canonical_bytes({"b": 1, "a": Path("x") / "y"}) == b'{"a":"x/y","b":1}'
    assert canonical_bytes(load_config(CONFIG)) == canonical_bytes(load_config(CONFIG))


def test_freeze_then_edit_prompt_raises_lock_mismatch(tmp_path: Path) -> None:
    from islands_harness.config import check_lock, freeze

    cfg = load_config(CONFIG)
    lock = freeze(cfg, REPO, frozen_on="2026-01-01", git_commit="deadbeef")
    check_lock(cfg, lock, REPO)
    with pytest.raises(LockMismatch):
        check_lock(cfg, lock.model_copy(update={"system_prompt": "0" * 64}), REPO)


# -- specs: the paired design ----------------------------------------------------------------


def _docs(n: int = 100) -> list[str]:
    return [f"inv-{i + 1:04d}" for i in range(n)]


def test_every_cell_pairs_index_to_same_document_and_seed() -> None:
    cfg = load_config(CONFIG)
    model = cfg.model_by_id("gpt-oss-20b-local")
    cells = [c for c in expand_cells(cfg, "A") if c.tools >= 2]
    assert len(cells) == 30
    reference = run_specs(cfg, model, cells[0], _docs())
    for cell in cells[1:]:
        specs = run_specs(cfg, model, cell, _docs())
        assert [(s.index, s.doc_id, s.epoch, s.sample_seed) for s in specs] == [
            (s.index, s.doc_id, s.epoch, s.sample_seed) for s in reference
        ]
        assert len(specs[0].tool_list) == cell.tools
        assert specs[0].tool_list[:2] == ("fetch_document", "submit_record")
    ids = {s.run_id for cell in cells for s in run_specs(cfg, model, cell, _docs())}
    assert len(ids) == 30 * 100


def test_rung1_offers_one_tool_and_runs_fewer() -> None:
    cfg = load_config(CONFIG)
    model = cfg.model_by_id("gpt-oss-20b-local")
    specs = run_specs(cfg, model, Cell(1, 0.0, "A"), _docs())
    assert len(specs) == cfg.sweep.rung1_runs_per_cell
    assert specs[0].tool_list == ("fetch_document",)


# -- faults -----------------------------------------------------------------------------


def _fault_config(**over: object) -> FaultConfig:
    base: dict[str, object] = {
        "kinds": ["timeout", "garbled_payload", "error_response", "empty_result"],
        "weights": [1, 1, 1, 1],
        "applies_to": "all_tools",
        "nested_across_rates": True,
        "simulate_delay_s": 0,
        "garble": GarbleConfig(char_fraction=0.2, truncate=True),
    }
    base.update(over)
    return FaultConfig(**base)  # type: ignore[arg-type]


def _injector(rate: float, doc: str = "inv-0007") -> FaultInjector:
    cfg = load_config(CONFIG)
    model = cfg.model_by_id("gpt-oss-20b-local")
    spec = run_specs(cfg, model, Cell(3, rate, "A"), _docs())[6]
    return FaultInjector(_fault_config(), spec, cfg.snapshot.root_seed)


def test_faults_are_reproducible_and_nested_across_rates() -> None:
    zero = _injector(0.0)
    assert all(zero.decide(i, "fetch_document") is None for i in range(200))
    by_rate = {
        r: [_injector(r).decide(i, "fetch_document") for i in range(400)]
        for r in (0.1, 0.2, 0.3, 0.4, 0.5)
    }
    assert by_rate[0.3] == [_injector(0.3).decide(i, "fetch_document") for i in range(400)]
    rates = sorted(by_rate)
    for lo, hi in pairwise(rates):
        for a, b in zip(by_rate[lo], by_rate[hi], strict=True):
            if a is not None:
                assert b == a  # once faulted, faulted at every higher rate, same kind
    fired = sum(k is not None for k in by_rate[0.5])
    assert 150 < fired < 250
    assert {k for k in by_rate[0.5] if k} == set(FaultKind)


def test_exempt_submit_option_never_faults_submit_record() -> None:
    inj = _injector(0.5)
    inj.config = _fault_config(applies_to="exempt_submit")
    assert all(inj.decide(i, "submit_record") is None for i in range(200))


def test_garble_never_json_never_original() -> None:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from islands_harness.faults import garble

    @settings(max_examples=200)
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=8), st.integers() | st.text(max_size=20), max_size=6
        ),
        st.integers(0, 10_000),
    )
    def check(payload: dict, i: int) -> None:
        text = json.dumps(payload, sort_keys=True)
        out = garble(text, _fault_config(), (20261006, "m", "A", "inv-0001", 0, i))
        assert out != text
        with pytest.raises(json.JSONDecodeError):
            json.loads(out)

    check()


# -- loop: a small in-test registry, so these tests do not depend on the task's tools ----------


def _mini_registry() -> Registry:
    async def fetch(args: dict, ctx: RunContext) -> str:
        return "INVOICE inv-0001 total 10.00 EUR"

    async def submit(args: dict, ctx: RunContext) -> str:
        ctx.record_submission(args["record"])
        return json.dumps({"status": "accepted"}, sort_keys=True)

    async def calc(args: dict, ctx: RunContext) -> str:
        return "42"

    obj = {"type": "object"}
    specs = {
        "fetch_document": ToolSpec(
            "fetch_document",
            "Fetch a document.",
            {**obj, "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
            fetch,
        ),
        "submit_record": ToolSpec(
            "submit_record",
            "Submit the record.",
            {**obj, "properties": {"record": {"type": "object"}}, "required": ["record"]},
            submit,
        ),
        "calculator": ToolSpec(
            "calculator",
            "Arithmetic.",
            {**obj, "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            calc,
        ),
    }
    return Registry(
        specs,
        required=["fetch_document", "submit_record"],
        mixes={"A": ["fetch_document", "submit_record", "calculator"]},
        rung1=["fetch_document"],
    )


def _agent(**over: object) -> AgentConfig:
    base: dict[str, object] = {
        "max_turns": 6,
        "max_tool_calls": 10,
        "token_limit": 100_000,
        "time_limit_s": 600,
        "parallel_tool_calls": "allow",
        "tool_choice": "auto",
        "strict_schemas": False,
        "on_stop_without_submit": "end",
        "on_malformed_arguments": "catalog_error",
    }
    base.update(over)
    return AgentConfig(**base)  # type: ignore[arg-type]


def _call(name: str, args: dict, i: int = 0) -> ToolCall:
    return ToolCall(
        id=f"c{i}", name=name, arguments=args, arguments_text=json.dumps(args, sort_keys=True)
    )


FETCH = _call("fetch_document", {"doc_id": "inv-0001"})
SUBMIT = _call("submit_record", {"record": {"invoice_number": "INV-1"}}, 1)
SAMPLING = Sampling(temperature=0.7, top_p=1.0, max_tokens=256, seed=1, thinking=None, effort=None)


async def _run(
    script: list[ScriptedTurn],
    *,
    rate: float = 0.0,
    kinds: list[str] | None = None,
    tools: int = 2,
    **agent: object,
):
    spec = RunSpec(
        run_id="r0",
        model_id="m",
        cell=Cell(tools, rate, "A"),
        index=0,
        doc_id="inv-0001",
        epoch=0,
        sample_seed=1,
        tool_list=("fetch_document", "submit_record"),
    )
    cfg = _fault_config(kinds=kinds, weights=[1] * len(kinds)) if kinds else _fault_config()
    provider = FakeProvider(script)
    ctx = RunContext(doc_id="inv-0001", document_text="", dataset_dir=Path("."), tools={})
    record, transcript = await run_one(
        spec,
        provider,
        _mini_registry(),
        FaultInjector(cfg, spec, 20261006),
        _agent(**agent),
        Prompts("SYSTEM PROMPT"),
        ctx=ctx,
        sampling=SAMPLING,
    )
    return record, transcript, provider, ctx


@pytest.mark.parametrize("kind", ["timeout", "error_response", "empty_result", "garbled_payload"])
async def test_submit_record_semantics_per_fault_kind(kind: str) -> None:
    """timeout, error_response and empty_result store no record (executed False); garbled
    stores it and garbles only the acknowledgement."""
    record, transcript, _, ctx = await _run(
        [ScriptedTurn(tool_calls=(SUBMIT,)), ScriptedTurn(text="giving up")], rate=1.0, kinds=[kind]
    )
    (event,) = record.fault_events
    assert event.kind == kind and event.tool == "submit_record"
    result = transcript[3].tool_results[0]
    if kind == "garbled_payload":
        assert event.executed and ctx.submissions == [{"invoice_number": "INV-1"}]
        assert record.outcome == Outcome.submitted and record.recovered is True
        assert result.content != json.dumps({"status": "accepted"}, sort_keys=True)
    else:
        assert not event.executed and ctx.submissions == []
        assert record.outcome == Outcome.no_submission and record.recovered is False


async def test_loop_reaches_every_termination_path_on_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_turn = ScriptedTurn(tool_calls=(FETCH,))
    cases: list[tuple[Outcome, list[ScriptedTurn], dict[str, object]]] = [
        (Outcome.submitted, [fetch_turn, ScriptedTurn(tool_calls=(SUBMIT,))], {}),
        (Outcome.no_submission, [ScriptedTurn(text="The total is 10 EUR.")], {}),
        (Outcome.limit_turns, [fetch_turn, fetch_turn], {"max_turns": 2}),
        (
            Outcome.limit_tool_calls,
            [ScriptedTurn(tool_calls=(FETCH, _call("fetch_document", {"doc_id": "x"}, 2)))],
            {"max_tool_calls": 1},
        ),
        (Outcome.limit_tokens, [fetch_turn], {"token_limit": 50}),
        (Outcome.refusal, [ScriptedTurn(stop_reason=StopReason.refusal)], {}),
        (
            Outcome.limit_output,
            [ScriptedTurn(tool_calls=(FETCH,), stop_reason=StopReason.max_tokens)],
            {},
        ),
        (Outcome.context_overflow, [ScriptedTurn(raise_overflow=True)], {}),
        (Outcome.aborted_transport, [ScriptedTurn(raise_transport=True)], {}),
    ]
    for expected, script, agent in cases:
        record, _, provider, _ = await _run(script, **agent)
        assert record.outcome == expected, (expected, record.outcome)
        assert provider.script == [], f"{expected}: unused scripted turns"
        if expected == Outcome.limit_output:  # a truncated turn's calls are never executed
            assert record.tool_calls == 0 and not any(e.kind == "tool_call" for e in record.events)

    class _Clock:  # seen only by loop.py, so the event loop keeps the real clock
        t = 0.0

        def monotonic(self) -> float:
            self.t += 100.0
            return self.t

    monkeypatch.setattr(loop_module, "time", _Clock())
    record, _, _, _ = await _run([fetch_turn], time_limit_s=50)
    assert record.outcome == Outcome.limit_time


async def test_loop_adds_nothing_of_its_own() -> None:
    """The transcript holds only system, user, assistant turns and tool results; the messages
    the FakeProvider received on call k equal the transcript prefix at call k."""
    bad = ToolCall(
        id="c9", name="fetch_document", arguments=None, arguments_text="{doc_id:", malformed=True
    )
    script = [
        ScriptedTurn(tool_calls=(bad,)),
        ScriptedTurn(text="checking", tool_calls=(FETCH,)),
        ScriptedTurn(tool_calls=(SUBMIT,)),
    ]
    record, transcript, provider, _ = await _run(script)
    assert record.outcome == Outcome.submitted and record.malformed_calls == 1
    assert [m.role for m in transcript] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert (
        transcript[0].content == "SYSTEM PROMPT"
        and transcript[1].content == "Document id: inv-0001"
    )
    assert all(m.content is None for m in transcript if m.role == "tool")
    assert transcript[3].tool_results[0].content == MALFORMED_ARGUMENTS_TEXT
    assert transcript[2].tool_calls[0].arguments_text == "{doc_id:"  # never repaired
    view = lambda ms: [(m.role, m.content, m.tool_calls, m.tool_results) for m in ms]  # noqa: E731
    for k, (sent, _tools, _sampling) in enumerate(provider.calls):
        assert view(sent) == view(transcript[: 2 + 2 * k])


def test_validate_arguments_minimal_rules() -> None:
    inner = {"type": "object", "properties": {"q": {"type": "number"}}, "required": ["q"]}
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "x": {"type": "number"},
            "obj": inner,
            "items": {"type": "array", "items": inner},
        },
        "required": ["n"],
    }
    assert validate_arguments(
        schema, {"n": 1, "x": 2.5, "obj": {"q": 1}, "items": [{"q": 1}], "extra": "ok"}
    )
    assert not validate_arguments(schema, {"x": 1})  # missing required at depth 0
    assert not validate_arguments(schema, {"n": True})  # bool is not an integer
    assert not validate_arguments(schema, {"n": 1.0})  # integer-valued float is not an integer
    assert not validate_arguments(
        schema, {"n": 1, "obj": {"q": "1"}}
    )  # depth-1 object: property type checked
    assert not validate_arguments(schema, {"n": 1, "obj": {}})  # depth-1 object: required checked
    assert not validate_arguments(
        schema, {"n": 1, "items": ["x"]}
    )  # depth-1 array: element type checked
    assert validate_arguments(
        schema, {"n": 1, "items": [{"q": "1"}]}
    )  # depth 2: not checked, by rule
    assert not validate_arguments(schema, [1])


def test_validate_arguments_on_the_public_submit_schema() -> None:
    task = Registry.load(
        REPO / "tasks" / "invoice_extraction" / "tools.py",
        required=["fetch_document", "submit_record"],
        mixes={"A": ["fetch_document", "submit_record"]},
        rung1=["fetch_document"],
    )
    schema = task.specs["submit_record"].parameters
    good = {
        "invoice_number": "INV-1",
        "invoice_date": "2026-01-02",
        "vendor_name": "V",
        "currency": "EUR",
        "total_amount": 10.0,
        "line_items": [],
    }
    assert validate_arguments(schema, {"record": good})
    assert not validate_arguments(
        schema, {"record": {**good, "total_amount": "10.00"}}
    )  # record field typed
    assert not validate_arguments(
        schema, {"record": {k: v for k, v in good.items() if k != "currency"}}
    )
    assert not validate_arguments(schema, {"rec": good})


# -- netguard -----------------------------------------------------------------------------


async def test_allowlist_transport_refuses_other_hosts() -> None:
    inner = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    transport = AllowlistTransport(["127.0.0.1:8081", "api.anthropic.com"], inner=inner)
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await client.get("http://127.0.0.1:8081/v1/models")).json() == {"ok": True}
        assert (await client.post("https://api.anthropic.com/v1/messages")).status_code == 200
        with pytest.raises(HostNotAllowed):
            await client.get("http://127.0.0.1:8001/v1/models")
        with pytest.raises(HostNotAllowed):
            await client.get("https://example.com/")


async def test_with_retry_logs_each_retry_and_exhausts() -> None:
    events: list[dict] = []
    calls = {"n": 0}

    async def always_503() -> None:
        calls["n"] += 1
        raise TransportError("503", retryable=True, status=503)

    policy = RetryPolicy(max_attempts=6, base_delay_s=0.0, max_delay_s=0.0)
    with pytest.raises(TransportExhausted):
        await with_retry(always_503, policy, events.append)
    assert calls["n"] == 6
    assert len(events) == 5 and all(e["event"] == "transport_retry" for e in events)

    async def bad_request() -> None:
        raise TransportError("400", retryable=False, status=400)

    with pytest.raises(TransportError):
        await with_retry(bad_request, policy, events.append)
    assert len(events) == 5


# -- storage and resume ---------------------------------------------------------------------


def test_existing_run_ids_ignores_partial_trailing_line(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.runs_path.write_bytes(b'{"run_id":"aaaa"}\n{"run_id":"bbbb"}\n{"run_id":"cc')
    assert storage.existing_run_ids() == {"aaaa", "bbbb"}


class _Responder:
    """A stateless scripted model: fetch when the last message is the user's, otherwise
    submit. ``fail_after`` makes the call after that many submits raise, standing in for a
    process killed mid-sweep."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.submits = 0
        self.fail_after = fail_after

    async def complete(self, messages, tools, sampling):  # noqa: ANN001, ANN201
        from islands_harness.providers.base import Message, Turn, Usage

        if self.fail_after is not None and self.submits >= self.fail_after:
            raise RuntimeError("simulated kill")
        doc = messages[1].content.split(": ", 1)[1]
        if messages[-1].role == "user":
            call = _call("fetch_document", {"doc_id": doc})
        else:
            call = _call("submit_record", {"record": {"invoice_number": doc}}, 1)
            self.submits += 1
        message = Message("assistant", None, tool_calls=[call])
        return Turn(
            "",
            [call],
            Usage(50, 10),
            StopReason.tool_calls,
            {"model": "responder"},
            message,
            {},
            {},
        )

    def identity(self) -> dict:
        return {"model": "responder"}


class _Grader:
    @staticmethod
    def score(submission, gold):  # noqa: ANN001, ANN205
        from types import SimpleNamespace

        ok = (
            isinstance(submission, dict)
            and submission.get("invoice_number") == gold["invoice_number"]
        )
        return SimpleNamespace(
            total=1.0 if ok else 0.0, per_field={"invoice_number": 1.0 if ok else 0.0}
        )


async def test_kill_and_resume_leaves_no_duplicate_or_gap(tmp_path: Path) -> None:
    from islands_harness.dataset import GoldRecord
    from islands_harness.runner import TaskBundle
    from islands_harness.runner import run_specs as drive

    cfg = load_config(CONFIG)
    model = cfg.model_by_id("gpt-oss-20b-local")
    docs = [f"inv-{i:04d}" for i in range(1, 6)]
    task = TaskBundle(
        registry=_mini_registry(),
        grader=_Grader,
        prompts=Prompts("SYSTEM PROMPT"),
        documents={d: f"INVOICE {d}" for d in docs},
        gold={d: GoldRecord(d, {"invoice_number": d}) for d in docs},
        dataset_dir=tmp_path,
    )
    cells = [Cell(2, 0.0, "A"), Cell(3, 0.2, "A"), Cell(2, 0.5, "A")]
    specs = [s for c in cells for s in run_specs(cfg, model, c, docs, n_runs=4)]
    storage = Storage(tmp_path / "out")
    common = {"config": cfg, "model": model, "tau": 0.8, "concurrency": 2}

    with pytest.raises(RuntimeError, match="simulated kill"):
        await drive(specs, _Responder(fail_after=5), task, storage, None, **common)
    before = storage.existing_run_ids()
    assert 0 < len(before) < len(specs)
    with storage.runs_path.open("ab") as f:  # a crash mid-write leaves a torn trailing line
        f.write(b'{"run_id": "torn-')

    summary = await drive(specs, _Responder(), task, storage, None, **common)
    lines = storage.runs_path.read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["run_id"] for line in lines]
    assert len(ids) == len(set(ids)) == len(specs)  # no duplicate
    assert set(ids) == {s.run_id for s in specs}  # no gap
    assert summary.skipped_existing == len(before)
    assert summary.executed == len(specs) - len(before)
    first = {s.run_id: s.index for s in specs if s.cell == cells[0]}
    order = [first[i] for i in ids if i in first]
    assert order == sorted(order)  # round robin: index order within a cell


async def test_resume_false_refuses_an_existing_runs_file(tmp_path: Path) -> None:
    from islands_harness.runner import run_specs as drive

    storage = Storage(tmp_path)
    storage.runs_path.write_text('{"run_id": "x"}\n', encoding="utf-8")
    cfg = load_config(CONFIG)
    with pytest.raises(FileExistsError):
        await drive(
            [],
            _Responder(),
            None,
            storage,
            None,
            config=cfg,
            model=cfg.model_by_id("gpt-oss-20b-local"),
            tau=0.8,
            resume=False,
        )  # type: ignore[arg-type]


class _FlakyResponder(_Responder):
    """Like _Responder, but the first submit turn for ``flaky_doc`` raises TransportExhausted
    once, so that run ends aborted_transport after one billed turn and is re-executed."""

    def __init__(self, flaky_doc: str) -> None:
        super().__init__()
        self.flaky_doc, self.tripped = flaky_doc, False

    async def complete(self, messages, tools, sampling):  # noqa: ANN001, ANN201
        from islands_harness.providers.netguard import TransportExhausted

        doc = messages[1].content.split(": ", 1)[1]
        if doc == self.flaky_doc and messages[-1].role != "user" and not self.tripped:
            self.tripped = True
            raise TransportExhausted("simulated transport failure")
        return await super().complete(messages, tools, sampling)


class _DeadServer(_Responder):
    """Like _Responder until ``down_after`` runs have submitted; then every call raises
    TransportExhausted, like a llama-server that died or hung mid-sweep."""

    def __init__(self, down_after: int) -> None:
        super().__init__()
        self.down_after = down_after

    async def complete(self, messages, tools, sampling):  # noqa: ANN001, ANN201
        from islands_harness.providers.netguard import TransportExhausted

        if self.submits >= self.down_after:
            raise TransportExhausted("simulated dead server")
        return await super().complete(messages, tools, sampling)


class _DeadDocs(_Responder):
    """Every call for the named documents raises TransportExhausted; the rest are served."""

    def __init__(self, dead: set[str]) -> None:
        super().__init__()
        self.dead = dead

    async def complete(self, messages, tools, sampling):  # noqa: ANN001, ANN201
        from islands_harness.providers.netguard import TransportExhausted

        if messages[1].content.split(": ", 1)[1] in self.dead:
            raise TransportExhausted("simulated transport failure")
        return await super().complete(messages, tools, sampling)


def _five_run_task(tmp_path: Path):  # noqa: ANN202
    from islands_harness.dataset import GoldRecord
    from islands_harness.runner import TaskBundle

    cfg = load_config(CONFIG)
    model = cfg.model_by_id("gpt-oss-20b-local")
    docs = [f"inv-{i:04d}" for i in range(1, 6)]
    task = TaskBundle(
        registry=_mini_registry(),
        grader=_Grader,
        prompts=Prompts("SYSTEM PROMPT"),
        documents={d: f"INVOICE {d}" for d in docs},
        gold={d: GoldRecord(d, {"invoice_number": d}) for d in docs},
        dataset_dir=tmp_path,
    )
    specs = run_specs(cfg, model, Cell(2, 0.0, "A"), docs, n_runs=5)
    common = {"config": cfg, "model": model, "tau": 0.8, "concurrency": 1}
    return task, specs, common


def _rows(storage: Storage) -> list[dict]:
    return [json.loads(x) for x in storage.runs_path.read_text(encoding="utf-8").splitlines()]


async def test_a_dead_server_halts_the_pass_and_records_none_of_its_runs(tmp_path: Path) -> None:
    from islands_harness.runner import TransportHalt
    from islands_harness.runner import run_specs as drive

    task, specs, common = _five_run_task(tmp_path)
    storage = Storage(tmp_path / "out")
    with pytest.raises(TransportHalt, match="twice in a row"):
        await drive(specs, _DeadServer(down_after=2), task, storage, None, **common)
    rows = _rows(storage)
    assert [r["index"] for r in rows] == [0, 1]  # the two runs before the server died
    assert all(r["outcome"] != "aborted_transport" for r in rows)

    summary = await drive(specs, _Responder(), task, storage, None, **common)  # server back
    rows = _rows(storage)
    assert sorted(r["index"] for r in rows) == [0, 1, 2, 3, 4]
    assert all(r["outcome"] != "aborted_transport" for r in rows)
    assert (summary.skipped_existing, summary.executed) == (2, 3)


async def test_an_isolated_double_abort_is_still_recorded(tmp_path: Path) -> None:
    from islands_harness.runner import run_specs as drive

    task, specs, common = _five_run_task(tmp_path)
    storage = Storage(tmp_path / "out")
    dead = {specs[1].doc_id, specs[4].doc_id}  # one mid-pass, one at the very end
    summary = await drive(specs, _DeadDocs(dead), task, storage, None, **common)
    rows = _rows(storage)
    assert [r["index"] for r in rows] == [0, 1, 2, 3, 4]
    aborted = [r["index"] for r in rows if r["outcome"] == "aborted_transport"]
    assert aborted == [1, 4] and summary.aborted_transport == 2
    assert all(r["success"] is None for r in rows if r["index"] in (1, 4))


class _RejectingServer(_Responder):
    """Every call for ``doc`` raises TransportExhausted with ``message``, as the provider does
    after its retries; the rest are served."""

    def __init__(self, doc: str, message: str) -> None:
        super().__init__()
        self.doc, self.message = doc, message

    async def complete(self, messages, tools, sampling):  # noqa: ANN001, ANN201
        from islands_harness.providers.netguard import TransportExhausted

        if messages[1].content.split(": ", 1)[1] == self.doc:
            raise TransportExhausted(self.message)
        return await super().complete(messages, tools, sampling)


@pytest.mark.parametrize(
    ("message", "cause"),
    [
        (
            'gave up after 6 attempts: HTTP 500: {"error":{"code":500,"message":"The model '
            'produced output that does not match the expected peg-native format"}}',
            "unparseable_model_output",
        ),
        ("gave up after 6 attempts: ConnectError: connection refused", "transport"),
    ],
)
async def test_an_aborted_run_records_why(tmp_path: Path, message: str, cause: str) -> None:
    from islands_harness.runner import abort_cause
    from islands_harness.runner import run_specs as drive

    task, specs, common = _five_run_task(tmp_path)
    storage = Storage(tmp_path / "out")
    await drive(specs, _RejectingServer(specs[2].doc_id, message), task, storage, None, **common)
    rows = {r["index"]: r for r in _rows(storage)}
    assert rows[2]["outcome"] == "aborted_transport"
    events = [e for e in rows[2]["events"] if e["kind"] == "abort_cause"]
    assert len(events) == 1 and events[0]["data"]["cause"] == cause == abort_cause(message)
    assert events[0]["data"]["error"] == message and events[0]["turn"] == 0
    assert all(e["kind"] != "abort_cause" for i in (0, 1, 3, 4) for e in rows[i]["events"])


async def test_reexecuted_abort_bills_both_attempts(tmp_path: Path) -> None:
    from islands_harness.dataset import GoldRecord
    from islands_harness.runner import SpendLedger, TaskBundle
    from islands_harness.runner import run_specs as drive

    cfg = load_config(CONFIG)
    model = cfg.model_by_id("gpt-oss-20b-local")
    docs = ["inv-0001", "inv-0002"]
    task = TaskBundle(
        registry=_mini_registry(),
        grader=_Grader,
        prompts=Prompts("SYSTEM PROMPT"),
        documents={d: f"INVOICE {d}" for d in docs},
        gold={d: GoldRecord(d, {"invoice_number": d}) for d in docs},
        dataset_dir=tmp_path,
    )
    specs = run_specs(cfg, model, Cell(2, 0.0, "A"), docs, n_runs=2)
    ledger = SpendLedger(prices={"input": 1.0}, cap_usd=None)  # 1 dollar per input token
    summary = await drive(
        specs,
        _FlakyResponder(specs[0].doc_id),
        task,
        Storage(tmp_path / "out"),
        ledger,
        config=cfg,
        model=model,
        tau=0.8,
    )
    rows = {
        json.loads(line)["run_id"]: json.loads(line)
        for line in (tmp_path / "out" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert summary.reexecuted == 1 and summary.aborted_transport == 0
    assert all(r["outcome"] == "submitted" and r["cost_usd"] == 100.0 for r in rows.values())
    # two clean runs at 2 turns x 50 input tokens, plus the aborted first attempt's one turn
    assert ledger.spent_usd == summary.spend_usd == 2 * 100.0 + 50.0


def test_spend_estimate_and_hosted_phase_counts() -> None:
    from islands_harness.provenance import Forecast, estimate, is_hosted, phase_counts
    from islands_harness.runner import load_prices

    cfg = load_config(CONFIG)
    hosted = load_models(REPO / "configs" / "hosted-models.yaml")[0]
    local = cfg.model_by_id("gpt-oss-20b-local")
    assert is_hosted(hosted) and not is_hosted(local)
    counts = phase_counts(cfg, hosted)
    assert (
        counts["sweep"] == (30 * 50 + 6 * 20) * len(cfg.sweep.mixes_to_run)
        and counts["noise_floor"] == 3 * 50 * 10
        and "determinism_gate" not in counts
    )
    assert phase_counts(cfg, local)["noise_floor"] == 50 * 20 + 50 * 100
    prices = load_prices(REPO / "configs" / "prices.v1.json", "claude-opus-5")
    assert prices == {
        "input": 5e-06,
        "output": 2.5e-05,
        "cache_read": 5e-07,
        "cache_write": 6.25e-06,
    }
    forecast = Forecast("opus5-hosted", [], 1000.0, 200.0, 4000.0, 3.0, "2026-09-25", 800.0)
    plan = estimate(forecast, prices, {"sweep": 100}, cap_usd=10.0)
    per_run = 1000 * 5e-06 + 200 * 2.5e-05 + 4000 * 5e-07 + 800 * 6.25e-06
    assert plan.usd["sweep"] == round(100 * per_run, 2) and plan.within_cap
    assert not estimate(forecast, prices, {"sweep": 1000}, cap_usd=10.0).within_cap
    with pytest.raises(KeyError):
        load_prices(REPO / "configs" / "prices.v1.json", "no-such-model")


async def test_factory_builds_the_hosted_provider_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The anthropic branch builds from the snapshot config without touching the network,
    refuses to build without an explicit key, and sends adaptive thinking with no fallbacks."""
    from islands_harness.providers.base import Message
    from islands_harness.providers.factory import build_provider

    cfg = load_config(CONFIG)
    hosted = load_models(REPO / "configs" / "hosted-models.yaml")[0]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        build_provider(hosted, cfg.agent)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-value")
    provider = build_provider(hosted, cfg.agent)
    try:
        body = provider.build_request(
            [Message("system", "SYSTEM"), Message("user", "Extract inv-0001")],
            _mini_registry().for_rung("A", 2),
            provider.sampling,
        )
        assert body["model"] == "claude-opus-5" and body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "medium"} and "fallbacks" not in body
        assert provider.identity()["fallbacks"] == "none"
    finally:
        await provider.aclose()
