"""Exploratory OpenRouter trial: the five models of configs/exploratory-openrouter.yaml on two
cells of mix A, all 100 documents each, in parallel. Not part of snapshot 1.

    .venv/Scripts/python.exe scripts/openrouter_trial.py [MODEL_ID ...]

Cells: 2 tools without faults (the task itself) and 6 tools at 30 percent faults (the task
under stress, with the calculator on offer). The local models ran both cells in snapshot 1's
sweep, so the comparison is direct.

The key is read from the workspace's secrets folder (../secrets/openrouter.txt, outside the
repository) into OPENROUTER_API_KEY for the child processes only; it is never printed or
logged. Each model has its spend cap in the config, and the runner stops at 95 percent of
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
KEY_FILE = ROOT.parent / "secrets" / "openrouter.txt"
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


def read_key() -> str:
    if not KEY_FILE.is_file():
        sys.exit(f"no key file at {KEY_FILE}")
    lines = [x.strip() for x in KEY_FILE.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if len(lines) != 1 or " " in lines[0]:
        sys.exit(f"{KEY_FILE} must hold the key alone on one line")
    return lines[0]


def main() -> int:
    env = {**os.environ, "PYTHONUTF8": "1", "OPENROUTER_API_KEY": read_key()}
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
