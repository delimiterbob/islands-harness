"""The local stack (M5): reading a llama-server's flags and /props against the config, the
weights hash cache, the doctor, and the three-regime determinism gate. Everything here is
offline: server answers are canned and the gate runs against the synthetic agent, so no test
touches a running server or the real weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from islands_harness import localstack, provenance
from islands_harness.config import load_config
from islands_harness.provenance import doctor, verify_determinism
from islands_harness.providers.synthetic import SyntheticAgent
from islands_harness.runner import load_task
from islands_harness.specs import gate_specs

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "snapshot-1.yaml"
GPT_ARGV = [
    r"C:\llama\llama-server.exe",
    "-m",
    r"C:\harness\models\weights\gpt-oss-20b-MXFP4.gguf",
    "--alias",
    "gpt-oss-20b",
    "--host",
    "127.0.0.1",
    "--port",
    "8081",
    "-c",
    "16384",
    "-np",
    "1",
    "--no-cache-prompt",
    "-ngl",
    "all",
    "-fa",
    "on",
    "-ctk",
    "f16",
    "-ctv",
    "f16",
    "-b",
    "2048",
    "-ub",
    "512",
    "--jinja",
    "--reasoning-format",
    "auto",
    "--no-webui",
]


@pytest.fixture(scope="module")
def cfg():  # noqa: ANN201
    return load_config(CONFIG)


def _props(**over):  # noqa: ANN003, ANN202
    props = {
        "build_info": "b11191-4b1a27fa0",
        "total_slots": 1,
        "default_generation_settings": {"n_ctx": 16384},
        "model_path": r"C:\harness\models\weights\gpt-oss-20b-MXFP4.gguf",
        "model_ftype": "MXFP4 MoE",
        "chat_template": "{{ '<|start|>' }} ... <|channel|> ... <|call|>",
        "chat_template_caps": {"supports_tool_calls": True},
    }
    props.update(over)
    return props


# -- flags and /props ---------------------------------------------------------------------


def test_parse_flags_reads_values_and_switches() -> None:
    flags = localstack.parse_flags(GPT_ARGV)
    assert flags["-np"] == "1" and flags["--no-cache-prompt"] is True and flags["-fa"] == "on"
    assert flags["-ngl"] == "all" and flags["--no-webui"] is True and flags["--port"] == "8081"
    assert localstack.ServerProcess(1, GPT_ARGV, flags).port == 8081


def test_flag_differences_catch_each_changed_flag(cfg) -> None:  # noqa: ANN001
    model = cfg.model_by_id("gpt-oss-20b-local")
    good = localstack.ServerProcess(1, GPT_ARGV, localstack.parse_flags(GPT_ARGV))
    assert localstack.flag_differences(good, model) == []
    assert localstack.flag_differences(good, model, flash_attn="off") == [
        "-fa: running 'on', expected 'off'"
    ]
    bad_argv = [a for a in GPT_ARGV if a != "--no-cache-prompt"]
    bad_argv[bad_argv.index("-np") + 1] = "4"
    bad = localstack.ServerProcess(1, bad_argv, localstack.parse_flags(bad_argv))
    diffs = localstack.flag_differences(bad, model)
    assert any(d.startswith("-np:") for d in diffs) and any("--no-cache-prompt" in d for d in diffs)


def test_props_differences(cfg) -> None:  # noqa: ANN001
    model = cfg.model_by_id("gpt-oss-20b-local")
    assert localstack.props_differences(_props(), model) == []
    diffs = localstack.props_differences(
        _props(build_info="b9999-deadbeef0", total_slots=4, chat_template="plain"), model
    )
    assert len(diffs) == 3 and any("GPT-OSS markers" in d for d in diffs)
    qwen = cfg.model_by_id("qwen3-14b-local")
    qwen_props = _props(
        model_path="x/Qwen3-14B-Q5_K_M.gguf",
        model_ftype="Q5_K - Medium",
        chat_template="... <tool_call> ... </tool_call> ...",
    )
    assert localstack.props_differences(qwen_props, qwen) == []


@pytest.mark.parametrize(
    "expected,ftype,ok",
    [
        ("Q5_K_M", "Q5_K - Medium", True),
        ("Q5_K_M", "Q5_K - Small", False),
        ("Q6_K", "Q6_K", True),
        ("MXFP4", "MXFP4 MoE", True),
        ("Q8_0", "Q4_K - Medium", False),
    ],
)
def test_quantization_names_match_llama_cpp_ftype(expected: str, ftype: str, ok: bool) -> None:
    assert localstack.quantization_matches(expected, ftype) is ok


# -- weights ------------------------------------------------------------------------------


def test_weights_hash_is_cached_until_the_file_changes(cfg, tmp_path: Path) -> None:  # noqa: ANN001
    import hashlib

    data = b"weights" * 1000
    (tmp_path / "models" / "weights").mkdir(parents=True)
    (tmp_path / "models" / "weights" / "w.gguf").write_bytes(data)
    lock = {
        "repo": "r",
        "revision": "x",
        "local_dir": "models/weights",
        "files": [
            {"path": "w.gguf", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        ],
    }
    (tmp_path / "models" / "w.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    model = cfg.model_by_id("gpt-oss-20b-local").model_copy(
        update={"lock": Path("models/w.lock.json")}
    )
    ok, detail = localstack.verify_weights(tmp_path, model)
    assert ok and "recomputed" in detail
    ok, detail = localstack.verify_weights(tmp_path, model)
    assert ok and "unchanged since" in detail
    (tmp_path / "models" / "weights" / "w.gguf").write_bytes(data[:-1] + b"X")
    ok, detail = localstack.verify_weights(tmp_path, model)
    assert not ok and "sha256" in detail


# -- doctor -------------------------------------------------------------------------------


def test_doctor_blocks_on_a_mismatched_server_and_never_contacts_a_hosted_api(
    cfg,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        localstack,
        "gpu_memory",
        lambda: {"name": "GPU", "driver": "1", "total_mib": 16303, "used_mib": 0},
    )
    monkeypatch.setattr(
        localstack, "verify_weights", lambda root, model, rehash=False: (True, "ok")
    )
    monkeypatch.setattr(localstack, "weights_path", lambda root, model: CONFIG)
    # The llama.cpp install is a property of the machine, not of this test: stubbed like the GPU, so the test
    # passes on a machine without it (a CI runner) as on the workstation.
    monkeypatch.setattr(
        provenance,
        "_check_llama_build",
        lambda config: provenance._check(
            "llama.cpp build", True, "installed (stubbed for the test)"
        ),
    )
    state = {"gpt-oss-20b-local": (True, [], {}), "qwen3-14b-local": (False, ["down"], {})}
    monkeypatch.setattr(
        localstack,
        "server_differences",
        lambda model, flash_attn=None: state.get(model.id, (False, ["down"], {})),
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    contacted: list[str] = []
    monkeypatch.setattr(localstack, "fetch", lambda model, path, **k: contacted.append(model.id))
    report = doctor(cfg, REPO)
    names = {c["name"]: c for c in report.checks}
    assert not report.blocking
    assert (
        names["qwen3-14b-local server"]["ok"] is False
        and names["qwen3-14b-local server"]["blocking"] is False
    )
    assert not any(n.endswith(" key") for n in names) and contacted == []  # no hosted model
    state["gpt-oss-20b-local"] = (True, ["-np: running '4', expected '1'"], {})
    report = doctor(cfg, REPO, model_id="gpt-oss-20b-local")
    assert report.blocking and "api.anthropic.com" not in report.outbound_hosts


# -- the gate -----------------------------------------------------------------------------


def test_gate_specs_are_the_anchor_cells_first_indices(cfg) -> None:  # noqa: ANN001
    model = cfg.model_by_id("gpt-oss-20b-local")
    specs = gate_specs(cfg, model, sorted(load_task(cfg, REPO).documents), "A")
    assert len(specs) == cfg.determinism_gate.specs
    assert {(s.cell.tools, s.cell.fault_rate, s.phase) for s in specs} == {(2, 0.0, "gate")}
    assert [s.index for s in specs] == list(range(len(specs))) and len(
        {s.doc_id for s in specs}
    ) == len(specs)


def _agent(cfg):  # noqa: ANN001, ANN202
    task = load_task(cfg, REPO)
    return SyntheticAgent(
        {d: g.fields for d, g in task.gold.items()}, task.documents, model_id="gpt-oss-20b-local"
    )


def test_gate_passes_on_a_deterministic_agent(cfg, tmp_path: Path) -> None:  # noqa: ANN001
    model = cfg.model_by_id("gpt-oss-20b-local")
    result = verify_determinism(cfg, model, _agent(cfg), repo_root=REPO, out_dir=tmp_path)
    gate = cfg.determinism_gate
    assert result.passed and result.label == "deterministic_verified"
    assert result.executions == gate.specs * gate.replays_per_regime * 3
    assert all(len(h) == gate.replays_per_regime * 3 for h in result.hashes.values())
    assert result.filler_requests == gate.filler_requests and result.filler_failures == 0
    record = json.loads((tmp_path / "determinism.json").read_text(encoding="utf-8"))
    assert (
        record["label"] == "deterministic_verified"
        and record["attempts"][0]["variant"] == "default"
    )
    executions = (tmp_path / "gate" / "default" / "executions.jsonl").read_text(encoding="utf-8")
    assert executions.count("\n") == result.executions


def test_gate_fails_on_drift_and_records_both_attempts(cfg, tmp_path: Path) -> None:  # noqa: ANN001
    model = cfg.model_by_id("gpt-oss-20b-local")

    class Drifting(SyntheticAgent):
        """Answers one document differently after the serial regime: a stack whose output
        depends on what ran before it."""

        calls = 0

        async def complete(self, messages, tools, sampling):  # noqa: ANN001, ANN201
            Drifting.calls += 1
            turn = await super().complete(messages, tools, sampling)
            user = next(m for m in messages if m.role == "user")
            if Drifting.calls > 60 and str(user.content).endswith(target) and turn.tool_calls:
                call = turn.tool_calls[0]
                if call.name == "submit_record":
                    return self._text(messages, 1, target, "Done.")
            return turn

    agent = _agent(cfg)
    target = gate_specs(cfg, model, sorted(agent.documents), "A")[3].doc_id
    drifting = Drifting(agent.gold, agent.documents, model_id=model.id)
    result = verify_determinism(cfg, model, drifting, repo_root=REPO, out_dir=tmp_path)
    assert not result.passed and result.label == "non_deterministic"
    assert len(result.disagreeing_specs) == 1
    divergence = result.divergences[0]
    assert divergence["distinct_transcripts"] == 2 and divergence["first_divergent_message"] == 4
    again = verify_determinism(
        cfg, model, _agent(cfg), repo_root=REPO, out_dir=tmp_path, variant="flash_attn_off"
    )
    record = json.loads((tmp_path / "determinism.json").read_text(encoding="utf-8"))
    assert again.passed and [a["variant"] for a in record["attempts"]] == [
        "default",
        "flash_attn_off",
    ]
    assert record["attempts"][0]["passed"] is False and record["label"] == "deterministic_verified"
