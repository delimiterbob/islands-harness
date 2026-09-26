"""Run the noise floor for every local model, one after another, unattended.

For each model: skip it when its floor is already complete (``noise-floor --analyze-only``
succeeds); otherwise stop any llama-server, start this model's pinned server, run
``islands noise-floor`` (which checks the server against the config and resumes run by
run), and stop the server. A failed floor is retried once on a restarted server. Fastest
model first, so a problem shows up early. While it runs, Windows is asked not to sleep
(SetThreadExecutionState, released when this process exits); no setting is changed.

    .venv/Scripts/python.exe scripts/noise_floor_all.py [MODEL_ID ...]

Everything is appended to results/s1/floor-driver.log. The last line is "DRIVER FINISHED"
with the models that completed and any that failed. Safe to rerun after an interruption.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISLANDS = ROOT / ".venv" / "Scripts" / "islands.exe"
LOG = ROOT / "results" / "s1" / "floor-driver.log"
# (serve name, model id), fastest first: about 2, 5 and 7 seconds a run at level 4.
MODELS = [
    ("qwen3-4b", "qwen3-4b-local"),
    ("qwen3-14b", "qwen3-14b-local"),
    ("gpt-oss-20b", "gpt-oss-20b-local"),
]


def log(line: str) -> None:
    stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"{stamp} {line}\n")


def islands(*args: str) -> int:
    """Run one islands command, streaming its output into the log; return its exit code."""
    log("$ islands " + " ".join(args))
    env = {**os.environ, "PYTHONUTF8": "1"}
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        proc = subprocess.run(
            [str(ISLANDS), *args],
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"exit {proc.returncode}")
    return proc.returncode


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def floor_for(serve_name: str, model_id: str) -> bool:
    if islands("noise-floor", "--model", model_id, "--analyze-only") == 0:
        log(f"{model_id}: floor already complete")
        return True
    for attempt in (1, 2):
        islands("serve", "stop")
        if islands("serve", "start", serve_name) != 0:
            log(f"{model_id}: server did not start (attempt {attempt})")
            continue
        code = islands("noise-floor", "--model", model_id)
        islands("serve", "stop")
        if code == 0:
            return True
        log(f"{model_id}: noise floor exited {code} (attempt {attempt})")
    return False


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    keep_awake()
    wanted = sys.argv[1:]
    models = [m for m in MODELS if not wanted or m[1] in wanted]
    log(f"DRIVER STARTED pid {os.getpid()}: {[m[1] for m in models]}")
    done, failed = [], []
    for serve_name, model_id in models:
        (done if floor_for(serve_name, model_id) else failed).append(model_id)
    log(f"DRIVER FINISHED: completed {done}, failed {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
