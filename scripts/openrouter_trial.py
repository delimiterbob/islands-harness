"""Exploratory OpenRouter trial: the five models of configs/exploratory-openrouter.yaml on two
cells of mix A, all 100 documents each, in parallel. Not part of snapshot 1.

    .venv/Scripts/python.exe scripts/openrouter_trial.py [MODEL_ID ...]

Cells: 2 tools without faults (the task itself) and 6 tools at 30 percent faults (the task
under stress, with the calculator on offer). The local models ran both cells in snapshot 1's
sweep, so the comparison is direct.

The keys are read from the workspace's secrets file (../secrets/.env, NAME=value lines,
outside the repository) into the child processes' environment only; OPENROUTER_API_KEY must
be set there. They are never printed or logged. Each model has its spend cap in the config, and the runner stops at 95 percent of
it. Output goes to results/exploratory/sweep/<model>/, logs to
results/exploratory/openrouter-trial/<model>.log. Safe to rerun: runs resume by id.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT.parent / "secrets" / ".env"
ISLANDS = ROOT / ".venv" / "Scripts" / "islands.exe"
CONFIG = "configs/exploratory-openrouter.yaml"
CELLS = ["2:0.0", "6:0.3"]
MODELS = [
    "gpt-oss-120b-or",
    "qwen3-235b-instruct-or",
    "qwen3-235b-thinking-or",
    "gpt-5.6-luna-or",
    "deepseek-v4.1-flash-or",
]


def read_secrets() -> dict[str, str]:
    """The NAME=value lines of the workspace's secrets file; comments and empty values are
    skipped. Nothing read here is ever printed."""
    if not SECRETS.is_file():
        sys.exit(f"no secrets file at {SECRETS}")
    values: dict[str, str] = {}
    for raw in SECRETS.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            values[name.strip()] = value
    return values


def main() -> int:
    secrets = read_secrets()
    if "OPENROUTER_API_KEY" not in secrets:
        sys.exit(f"OPENROUTER_API_KEY has no value in {SECRETS}")
    env = {**os.environ, **secrets, "PYTHONUTF8": "1"}
    logs = ROOT / "results" / "exploratory" / "openrouter-trial"
    logs.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODELS if len(sys.argv) < 2 or m in sys.argv[1:]]
    started = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    procs = {}
    for model in models:
        log = (logs / f"{model}.log").open("a", encoding="utf-8", newline="\n")
        log.write(f"--- {started} islands sweep {model} cells {CELLS}\n")
        log.flush()
        procs[model] = subprocess.Popen(
            [
                str(ISLANDS),
                "sweep",
                "-c",
                CONFIG,
                "--exploratory",
                "--model",
                model,
                "--mix",
                "A",
                "--cells",
                *CELLS,
                "--resume",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    codes = {model: proc.wait() for model, proc in procs.items()}
    print(f"openrouter trial started {started}; exit codes {codes}")
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
