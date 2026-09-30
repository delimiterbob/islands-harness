"""The local llama.cpp stack: find the running server, read what it reports, compare it with
the config's ``expect`` block, verify the weights, and start or stop the server
(ARCHITECTURE.md D9, D10, D12 and Section 8).

Two sources describe a running llama-server, and neither is enough alone:

- ``GET /props`` reports the build, the slot count, the context size, the weights path, the
  model's quantization type and the chat template, but not flash attention, the KV-cache
  types, the batch sizes or whether prompt caching is off.
- The server process's own command line carries exactly those flags, because
  ``server/llamacpp/serve.ps1`` passes every one explicitly.

``server_differences`` checks both against the model's ``expect`` block. Every HTTP request
goes to the model's own loopback address through the netguard allowlist.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from islands_harness.config import ModelConfig, SnapshotConfig

# Flags serve.ps1 passes without a value; every other flag it passes takes one.
_BOOLEAN_FLAGS = {"--no-cache-prompt", "--jinja", "--no-webui"}
# Chat-template markers of the tool-call format each expect.chat_format names.
TEMPLATE_MARKERS = {
    "GPT-OSS": ["<|start|>", "<|channel|>", "<|call|>"],
    "Hermes 2 Pro": ["<tool_call>", "</tool_call>"],
}
_QUANT_SUFFIX = {"S": "Small", "M": "Medium", "L": "Large"}
VERIFIED_CACHE = ".verified.json"


# --------------------------------------------------------------------------------------
# The running server
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerProcess:
    pid: int
    argv: list[str]
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def port(self) -> int | None:
        value = self.flags.get("--port")
        return int(value) if isinstance(value, str) and value.isdigit() else None


def parse_flags(argv: list[str]) -> dict[str, Any]:
    """llama-server arguments as {flag: value}, boolean flags as True."""
    flags: dict[str, Any] = {}
    i = 1  # argv[0] is the executable
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("-"):
            if arg in _BOOLEAN_FLAGS or i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                flags[arg] = True
                i += 1
            else:
                flags[arg] = argv[i + 1]
                i += 2
        else:
            i += 1
    return flags


def server_processes() -> list[ServerProcess]:
    """Every running llama-server with its command line. Windows reads Win32_Process through
    PowerShell; elsewhere ``ps``. An empty list when none runs or the listing fails."""
    rows: list[tuple[int, str]] = []
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | "
                    "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
            data = json.loads(out) if out else []
            for item in data if isinstance(data, list) else [data]:
                if item.get("CommandLine"):
                    rows.append((int(item["ProcessId"]), str(item["CommandLine"])))
        else:
            out = subprocess.run(
                ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=30, check=True
            ).stdout
            for line in out.splitlines():
                pid, _, args = line.strip().partition(" ")
                if "llama-server" in args.split(" ", 1)[0]:
                    rows.append((int(pid), args))
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    processes = []
    for pid, command in rows:
        argv = shlex.split(command, posix=True)
        processes.append(ServerProcess(pid, argv, parse_flags(argv)))
    return processes


def model_port(model: ModelConfig) -> int | None:
    return urlsplit(model.base_url or "").port


def server_for(
    model: ModelConfig, processes: list[ServerProcess] | None = None
) -> ServerProcess | None:
    """The llama-server process listening on the model's configured port, if any."""
    port = model_port(model)
    return next(
        (p for p in (processes if processes is not None else server_processes()) if p.port == port),
        None,
    )


def expected_flags(model: ModelConfig, *, flash_attn: str | None = None) -> dict[str, Any]:
    """The flags serve.ps1 must have passed for this model, from its expect block."""
    e = model.expect
    if e is None:
        raise ValueError(f"model {model.id} has no expect block")
    return {
        "--host": "127.0.0.1",
        "--port": str(model_port(model)),
        "-c": str(e.ctx_size),
        "-np": str(e.parallel_slots),
        "--no-cache-prompt": True,
        "-fa": flash_attn or e.flash_attn,
        "-ctk": e.kv_cache_type,
        "-ctv": e.kv_cache_type,
        "-b": str(e.batch_size),
        "-ub": str(e.ubatch_size),
        "--jinja": True,
    }


def flag_differences(
    process: ServerProcess, model: ModelConfig, *, flash_attn: str | None = None
) -> list[str]:
    """Where the running server's command line departs from the expect block."""
    out = []
    for flag, want in expected_flags(model, flash_attn=flash_attn).items():
        got = process.flags.get(flag)
        if got != want:
            out.append(f"{flag}: running {got!r}, expected {want!r}")
    # The server reports Windows paths; PureWindowsPath splits them on any system.
    weights = PureWindowsPath(str(process.flags.get("-m", ""))).name
    if model.expect is not None and weights != model.expect.weights_file:
        out.append(f"-m: running {weights!r}, expected {model.expect.weights_file!r}")
    return out


def quantization_matches(expected: str, ftype: str) -> bool:
    """Compare a quantization name (Q5_K_M, MXFP4) with llama.cpp's model_ftype text
    ("Q5_K - Medium", "MXFP4 MoE")."""
    head, _, suffix = expected.rpartition("_")
    if head and suffix in _QUANT_SUFFIX:
        return ftype.startswith(head) and _QUANT_SUFFIX[suffix] in ftype
    return expected in ftype


def props_differences(props: dict[str, Any], model: ModelConfig) -> list[str]:
    """Where the running server's GET /props departs from the expect block."""
    e = model.expect
    if e is None:
        return [f"model {model.id} has no expect block"]
    out = []
    build = f"{e.build}-{e.build_commit[:9]}"
    if props.get("build_info") != build:
        out.append(f"build_info {props.get('build_info')!r}, expected {build!r}")
    if props.get("total_slots") != e.parallel_slots:
        out.append(f"total_slots {props.get('total_slots')}, expected {e.parallel_slots}")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    if n_ctx != e.ctx_size:
        out.append(f"n_ctx {n_ctx}, expected {e.ctx_size}")
    path = PureWindowsPath(str(props.get("model_path", ""))).name
    if path != e.weights_file:
        out.append(f"model_path {path!r}, expected {e.weights_file!r}")
    ftype = str(props.get("model_ftype", ""))
    if not quantization_matches(e.quantization, ftype):
        out.append(f"model_ftype {ftype!r} does not match quantization {e.quantization!r}")
    template = str(props.get("chat_template", ""))
    missing = [m for m in TEMPLATE_MARKERS.get(e.chat_format, []) if m not in template]
    if e.chat_format not in TEMPLATE_MARKERS:
        out.append(f"no template markers known for chat_format {e.chat_format!r}")
    elif missing:
        out.append(f"chat template lacks the {e.chat_format} markers {missing}")
    if not (props.get("chat_template_caps") or {}).get("supports_tool_calls"):
        out.append("chat template does not declare tool-call support")
    return out


async def _get_json(model: ModelConfig, path: str, timeout_s: float) -> tuple[int, Any]:
    from islands_harness.providers.netguard import build_http_client

    parts = urlsplit(model.base_url or "")
    url = f"{parts.scheme}://{parts.netloc}{path}"
    async with build_http_client(model.allowed_hosts, timeout_s) as client:
        response = await client.get(url)
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, body


def fetch(model: ModelConfig, path: str, *, timeout_s: float = 5.0) -> tuple[int, Any] | None:
    """GET a path on the model's server (allowlisted); None when it does not answer."""
    try:
        return asyncio.run(_get_json(model, path, timeout_s))
    except Exception:  # noqa: BLE001 - any failure to reach the server means "not running"
        return None


def is_healthy(model: ModelConfig) -> bool:
    result = fetch(model, "/health")
    return result is not None and result[0] == 200


def server_differences(
    model: ModelConfig, *, flash_attn: str | None = None
) -> tuple[bool, list[str], dict[str, Any]]:
    """(running, differences, evidence) for the model's server: /props and the process's
    command line against the expect block. ``evidence`` keeps what was read, for the gate's
    record."""
    result = fetch(model, "/props")
    if result is None or result[0] != 200 or not isinstance(result[1], dict):
        return False, ["the server is not answering on " + str(model.base_url)], {}
    props = result[1]
    diffs = props_differences(props, model)
    process = server_for(model)
    if process is None:
        diffs.append("no llama-server process found for this port, so its flags were not checked")
    else:
        diffs.extend(flag_differences(process, model, flash_attn=flash_attn))
    evidence = {
        "build_info": props.get("build_info"),
        "total_slots": props.get("total_slots"),
        "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
        "model_path": props.get("model_path"),
        "model_ftype": props.get("model_ftype"),
        "chat_template_sha256": hashlib.sha256(
            str(props.get("chat_template", "")).encode("utf-8")
        ).hexdigest(),
        "command_line": process.argv if process is not None else None,
    }
    return True, diffs, evidence


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------


def model_lock(repo_root: Path, model: ModelConfig) -> dict[str, Any]:
    if model.lock is None:
        raise ValueError(f"model {model.id} has no lock file")
    return json.loads((Path(repo_root) / model.lock).read_text(encoding="utf-8"))


def weights_path(repo_root: Path, model: ModelConfig) -> Path:
    lock = model_lock(repo_root, model)
    return Path(repo_root) / lock.get("local_dir", "models/weights") / lock["files"][0]["path"]


def sha256_large(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(
    repo_root: Path, model: ModelConfig, *, rehash: bool = False
) -> tuple[bool, str]:
    """(ok, detail) for the model's weights against its lock: size, then sha256. A verified
    hash is cached beside the weights keyed by size and modification time, so a file that has
    not changed is not re-read (about half a minute per file); ``rehash`` ignores the cache."""
    lock = model_lock(repo_root, model)
    entry = lock["files"][0]
    path = weights_path(repo_root, model)
    if not path.is_file():
        return False, f"{path.name} is missing; see README step 4"
    stat = path.stat()
    if stat.st_size != int(entry["size_bytes"]):
        return False, f"{path.name} is {stat.st_size} bytes, the lock says {entry['size_bytes']}"
    cache_path = path.parent / VERIFIED_CACHE
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    hit = cache.get(path.name)
    if (
        not rehash
        and hit
        and hit.get("size") == stat.st_size
        and hit.get("mtime_ns") == stat.st_mtime_ns
        and hit.get("sha256") == entry["sha256"]
    ):
        return (
            True,
            f"{path.name} matches its lock (hash verified {hit['verified_on']}, file unchanged since)",
        )
    actual = sha256_large(path)
    if actual != entry["sha256"]:
        return (
            False,
            f"{path.name} sha256 {actual[:12]}..., the lock says {entry['sha256'][:12]}...",
        )
    cache[path.name] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": actual,
        "verified_on": datetime.date.today().isoformat(),
    }
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True, f"{path.name} matches its lock (sha256 recomputed)"


def fetch_weights(repo_root: Path, model: ModelConfig) -> Path:
    """Download the model's weights at the pinned revision with huggingface_hub, unless the
    file is already present. Verification is separate (verify_weights)."""
    from huggingface_hub import hf_hub_download

    lock = model_lock(repo_root, model)
    target = weights_path(repo_root, model)
    if target.is_file():
        return target
    hf_hub_download(
        repo_id=lock["repo"],
        filename=lock["files"][0]["path"],
        revision=lock["revision"],
        local_dir=str(target.parent),
    )
    return target


# --------------------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------------------


def gpu_memory() -> dict[str, Any] | None:
    """Name, driver and memory (MiB) of GPU 0 from nvidia-smi, or None without one."""
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, driver, total, used = (x.strip() for x in out[0].split(","))
    return {"name": name, "driver": driver, "total_mib": int(total), "used_mib": int(used)}


# --------------------------------------------------------------------------------------
# Start and stop
# --------------------------------------------------------------------------------------


def resolve_local_model(config: SnapshotConfig, name: str) -> ModelConfig:
    """A local model by config id (gpt-oss-20b-local) or serve name (gpt-oss-20b)."""
    for m in config.models:
        if m.expect is not None and name in (m.id, m.model):
            return m
    raise KeyError(f"no local model named {name!r} in the config")


def serve_start(
    config: SnapshotConfig,
    name: str,
    repo_root: Path,
    *,
    flash_attn: str = "on",
    skip_hash_check: bool = False,
    wait_s: float = 300.0,
) -> tuple[ModelConfig, Path]:
    """Start serve.ps1 for one model in the background, log to results/logs, and wait until
    /health answers. Refuses when any llama-server already runs (the GPU holds one model)."""
    model = resolve_local_model(config, name)
    running = server_processes()
    if running:
        ports = sorted(p.port for p in running if p.port is not None)
        raise RuntimeError(
            f"a llama-server is already running (ports {ports}); run `islands serve stop` first"
        )
    log_dir = Path(repo_root) / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"llama-server-{model.model}.log"
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(Path(repo_root) / "server" / "llamacpp" / "serve.ps1"),
        "-Model",
        model.model,
        "-FlashAttn",
        flash_attn,
    ]
    if skip_hash_check:
        command.append("-SkipHashCheck")
    flags = 0
    if sys.platform == "win32":
        # A new process group without a visible window. Not DETACHED_PROCESS: PowerShell
        # without a console exits at once, silently, and the server never starts. The
        # server outlives `uv run` either way (checked on 2026-09-26).
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    with log.open("ab") as out:
        launcher = subprocess.Popen(  # noqa: S603
            command, cwd=repo_root, stdout=out, stderr=subprocess.STDOUT, creationflags=flags
        )
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if is_healthy(model):
            return model, log
        if launcher.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(
                f"serve.ps1 exited with code {launcher.returncode} before {model.id} became "
                f"healthy; the end of {log}:\n{tail}"
            )
        time.sleep(1.0)
    raise TimeoutError(f"{model.id} did not become healthy within {wait_s:.0f} s; see {log}")


def serve_stop(config: SnapshotConfig, name: str | None = None) -> list[int]:
    """Stop the llama-server of one model, or every llama-server. Returns the stopped pids."""
    processes = server_processes()
    if name is not None:
        port = model_port(resolve_local_model(config, name))
        processes = [p for p in processes if p.port == port]
    stopped = []
    for p in processes:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, check=False
            )
        else:
            os.kill(p.pid, 15)
        stopped.append(p.pid)
    return stopped
