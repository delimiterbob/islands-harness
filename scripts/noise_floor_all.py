"""Run the noise floor for every local model, one after another, unattended.

For each model: skip it when its floor is already complete (``noise-floor --analyze-only``
succeeds); otherwise stop any llama-server, start this model's pinned server, run
``islands noise-floor`` (which checks the server against the config and resumes run by
run), and stop the server. Fastest model first, so a problem shows up early.

Recovery. A watchdog kills the floor when no run has landed for STALL_MINUTES of wall-clock
time, which is also what a hibernation looks like once the machine wakes; the runner itself
stops when two runs in a row cannot reach the server (runner.TransportHalt). Either way the
driver restarts the server and resumes, and gives up on a model only after MAX_FAILURES
attempts in a row that added no run.

Sleep. While it runs, Windows is asked not to sleep when idle (SetThreadExecutionState,
released when this process exits; no setting is changed). No program can veto a sleep or
hibernation chosen from the Start menu; the watchdog resumes the floor after the wake.

    .venv/Scripts/python.exe scripts/noise_floor_all.py [MODEL_ID ...]

Everything is appended to results/s1/floor-driver.log. The last line is "DRIVER FINISHED"
with the models that completed and any that failed. Safe to rerun after an interruption.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISLANDS = ROOT / ".venv" / "Scripts" / "islands.exe"
LOG = ROOT / "results" / "s1" / "floor-driver.log"
STALL_MINUTES = 20  # the slowest floor run takes well under a minute
MAX_FAILURES = 5
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


def islands(*args: str, watch: Path | None = None) -> int:
    """Run one islands command, streaming its output into the log; return its exit code.
    With ``watch``, kill the command (and its children) when that file stops growing for
    STALL_MINUTES of wall-clock time."""
    log("$ islands " + " ".join(args))
    env = {**os.environ, "PYTHONUTF8": "1"}
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        proc = subprocess.Popen(
            [str(ISLANDS), *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT
        )
        size, changed = _size(watch), time.time()
        while proc.poll() is None:
            time.sleep(30)
            if watch is None:
                continue
            now_size = _size(watch)
            if now_size != size:
                size, changed = now_size, time.time()
            elif time.time() - changed > STALL_MINUTES * 60:
                log(f"no new run for {STALL_MINUTES} minutes (a hibernation or a hung server)")
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
                proc.wait()
    log(f"exit {proc.returncode}")
    return proc.returncode


def _size(path: Path | None) -> int:
    try:
        return path.stat().st_size if path is not None else 0
    except OSError:
        return 0


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def floor_for(serve_name: str, model_id: str) -> bool:
    if islands("noise-floor", "--model", model_id, "--analyze-only") == 0:
        log(f"{model_id}: floor already complete")
        return True
    runs = ROOT / "results" / "s1" / model_id / "floor" / "runs.jsonl"
    failures = 0
    while failures < MAX_FAILURES:
        islands("serve", "stop")
        if islands("serve", "start", serve_name) != 0:
            failures += 1
            log(f"{model_id}: server did not start ({failures} failed attempts in a row)")
            continue
        before = _size(runs)
        code = islands("noise-floor", "--model", model_id, watch=runs)
        islands("serve", "stop")
        if code == 0:
            return True
        failures = 0 if _size(runs) > before else failures + 1
        log(
            f"{model_id}: floor stopped with exit {code}; relaunching ({failures} in a row without progress)"
        )
    log(f"{model_id}: giving up after {MAX_FAILURES} attempts without progress")
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
