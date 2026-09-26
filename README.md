# Islands Harness

The instrument behind [islandsofstability.com](https://islandsofstability.com). It runs one
AI agent through one fixed task thousands of times under controlled conditions and reports
the shape of the region where it works: a 6 by 6 sweep of tool count against injected
tool-fault rate, with sampling error and rerun jitter measured first, and one pre-registered
verdict per proposition.

The harness is the product. The map is its output. `ARCHITECTURE.md` holds the design and
every decision with its rejected alternative. `PREREGISTRATION.md` mirrors the decision
rules; `configs/preregistration/r4.yaml` is the machine-readable copy the analysis reads.
`docs/` holds the method note, the client-mode guide and the reproducibility recipe.

Status, 2026-09-26: M0 (loop, faults, adapter, freeze, selftest, smoke), M1 (dataset,
grader, tools), M2 (runner, spend ledger, probe, estimate, Anthropic adapter), M3
(statistics, verdicts, `islands analyze`), M4 (report, bundle, `verify`, `replay`,
`analyze --check`, schemas) and M5 (doctor, `serve`, `models verify`, the determinism gate)
are built. All three local stacks passed the three-regime determinism gate on 2026-09-26:
gpt-oss-20b, Qwen3-14B and Qwen3-4B each reproduced all 20 gate runs byte for byte across
15 executions. M6 is under way: `calibrate`, `freeze --dry-run` and `deposit` are built. The
pilot found every model at the task's ceiling on levels 1 to 3, so the task gained level 4
(amounts must be computed), an anchor study widened the calibration band to [0.15, 0.95],
and Qwen3-4B joined as a third replicate. Difficulty 4 at tau 0.8 makes all three
replicates, and the snapshot dataset is now level 4 (`docs/method-note.md` Sections 6 and
7). The author confirmed the pre-registration decision by decision, snapshot 1 runs the
three local models only, on mixes A and B, and the harness was frozen on
2026-09-26 (`configs/snapshot-1.lock.json`, tag v1.0.0, local). The Zenodo deposit
of `islands deposit` comes before any snapshot run; the public repository follows it. `islands selftest` now runs a whole synthetic
snapshot through the real pipeline and the reviewer's checks, and the mock site's Map page
renders the snapshot.json it produces. On gpt-oss-20b through llama.cpp, an exploratory
300-run sweep was killed after 14 runs and resumed to exactly the 300 planned runs, with no
duplicate and no gap. The statistics meet their coverage and false-survival bars on 3,000
simulated snapshots (`docs/method-note.md`). One M2 criterion is open: the 20-run hosted
dry run under a 5 dollar cap, which needs the owner's API key and go-ahead. M6 to M9
remain; each `TODO(M6)` to `TODO(M9)` marker follows the build plan in `ARCHITECTURE.md`
Section 16. Nothing has been frozen or measured.

## Install on Windows 11 (the author's workstation)

Run these in order in a regular PowerShell unless stated otherwise. Each command is the
exact one the design fixes; do not substitute.

1. uv, which installs Python and every pinned package:

       winget install --id=astral-sh.uv -e

   Then close and reopen the shell so `uv` is on the path.

2. Python 3.12 and the locked environment, from the clone:

       uv python install 3.12
       uv sync --locked

   `uv sync --locked` refuses to run if `uv.lock` disagrees with `pyproject.toml`. The lock
   is committed; regenerate it only in M0 with `uv lock`, never after the freeze.

3. llama.cpp, the pinned build b11191 with CUDA 13.4 (about 580 MB of downloads, each
   checked against its published SHA-256 before anything is extracted):

       powershell -ExecutionPolicy Bypass -File server\llamacpp\install.ps1

   It installs to `%LOCALAPPDATA%\islands\llama.cpp\b11191\` and prints `llama-server
   --version`. The CUDA 13.4 build relies on the driver's CUDA minor-version compatibility;
   this workstation's driver 591.86 reports CUDA 13.1. If llama-server fails with a CUDA
   error, update the NVIDIA driver first, and fall back to `-Cuda 12.4` only if that fails.

4. The two model files, pinned to exact repository commits (about 22.6 GB together; both
   repositories are Apache-2.0 and ungated, so no token is needed):

       uv run hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --revision ef9b12f2ff56c69cf32153a02784e7a3c88bf524 --local-dir models/weights
       uv run hf download Qwen/Qwen3-14B-GGUF Qwen3-14B-Q5_K_M.gguf --revision 530227a7d994db8eca5ab5ced2fb692b614357fd --local-dir models/weights
       uv run hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q5_K_M.gguf --revision bc640142c66e1fdd12af0bd68f40445458f3869b --local-dir models/weights
       uv run islands models fetch --verify

   `models/*.lock.json` holds each file's size and SHA-256; `models fetch --verify`
   recomputes them and the sweep refuses to start on a mismatch.

5. Git long paths, because transcripts nest deep:

       git config --global core.longpaths true

Everything runs natively on Windows; there is no WSL2, Docker or vLLM (D9). Start one model
server at a time, because the RTX 5080's 16 GB holds one:

    powershell -ExecutionPolicy Bypass -File server\llamacpp\serve.ps1 -Model gpt-oss-20b    # port 8081
    powershell -ExecutionPolicy Bypass -File server\llamacpp\serve.ps1 -Model qwen3-14b      # port 8082
    powershell -ExecutionPolicy Bypass -File server\llamacpp\serve.ps1 -Model qwen3-4b       # port 8083

The script checks the weights hash, then starts llama-server with one slot, prompt caching
off and every flag the config's `expect` block names (D12). Stop it with Ctrl+C. The harness
can do the same in the background, logging to `results/logs/`:

    uv run islands serve start gpt-oss-20b      # waits until /health answers
    uv run islands serve status
    uv run islands serve stop

`islands doctor` then checks the toolchain, the dataset lock, the llama.cpp install, the GPU,
each weights file against its lock (hashed once, then cached until the file changes), and
the running server: `GET /props` (build, slots, context, weights, quantization, chat
template) and the server process's own command line (every flag). A server that differs
from the config blocks every run command that checks it; a server that is not running is
only a warning, because one model runs at a time. Doctor never contacts a hosted API.

## Quickstart

    uv run islands selftest              # whole pipeline on the fake provider, no network
    uv run islands doctor -c configs/snapshot-1.yaml
    uv run islands smoke --model gpt-oss-20b-local -c configs/snapshot-1.yaml   # with its server running

Every command is `uv run islands <cmd>`; `uv run` syncs the locked environment first.
`islands --help` lists the subcommands and `ARCHITECTURE.md` Section 6 describes each.

The selftest writes every loop path's run to `results/selftest/`, then a whole synthetic
snapshot to `results/selftest/snapshot/`: 1,320 runs of a deterministic synthetic agent
(`providers/synthetic.py`) on 40 task documents, analysis, `snapshot.json`, heatmap,
manifest and bundle, followed by re-analysis, re-grading, four replays and the SHA256SUMS
check. Everything it produces is labelled as a selftest, not a measurement.

To see a snapshot on the mock site, copy its `snapshot.json` into `../mock/data/` and open
the Map page with the file named in the address; without the parameter the page shows the
empty grid:

    Copy-Item results\selftest\snapshot\snapshot.json ..\mock\data\selftest-snapshot.json
    node ..\mock\serve.js 5173
    # http://localhost:5173/map.html?snapshot=data/selftest-snapshot.json

The heatmap PNG (`islands report RESULTS --png`) needs the optional plots extra:
`uv sync --locked --extra plots`. The site itself uses the SVG.

## Snapshot 1, in order

    uv run islands doctor -c configs/snapshot-1.yaml
    uv run islands dataset generate --seed 20261006 --n 100 --difficulty 4
    uv run islands dataset verify
    uv run islands models fetch --verify
    uv run islands serve start gpt-oss-20b                     # or run serve.ps1 in its own window
    uv run islands smoke --model gpt-oss-20b-local
    uv run islands calibrate --model gpt-oss-20b-local         # pilot per difficulty level (repeat per model)
    uv run islands calibrate --recommend                       # one difficulty and tau for all models; write r4.yaml
    uv run islands verify-determinism --model gpt-oss-20b-local
    uv run islands freeze --dry-run                            # what would be locked, what is still unconfirmed
    uv run islands freeze                                      # point of no return: refuses while anything is unconfirmed
    uv run islands deposit                                     # the dated pre-registration deposit, for Zenodo
    uv run islands noise-floor --model gpt-oss-20b-local
    uv run islands sweep --model gpt-oss-20b-local --mix A --resume
    uv run islands analyze results/s1/gpt-oss-20b-local          # cells.csv, results.json, verdict.json, statement.txt
    uv run islands report results/s1/gpt-oss-20b-local
    uv run islands verify results/s1/gpt-oss-20b-local
    uv run islands bundle results/s1/gpt-oss-20b-local

Then stop that server, start `qwen3-14b`, and repeat from `smoke` for `qwen3-14b-local`;
then the same for `qwen3-4b` and `qwen3-4b-local`.
The determinism gate runs before the freeze, because a failed gate can change a server flag
(flash attention off), and that is a config change. Local runs are serial: at an estimated
2 to 6 seconds each (smoke on 2026-09-25, both models, 0 and 50 percent faults), one model's
snapshot (about 10,000 runs with the noise floor) is roughly 6 to 17 hours of unattended GPU
time; high fault rates add turns, so plan for the upper end. `sweep` is safe to kill and resumes by run id; it stops at 95 percent of
the spend cap and never prints a verdict. The deposit (r4.yaml, the lock, the forecast) is
dated before any full cell runs.

## The hosted model

Snapshot 1 runs the three local models only (author's decision, 2026-09-26). The hosted
model's definition lives in `configs/hosted-models.yaml`; copy its entry into a later
snapshot's config to include it, then run `probe` and `estimate` for it before the freeze.

`opus5-hosted` runs through the official Anthropic SDK and is optional. Nothing hosted runs
unless `ANTHROPIC_API_KEY` is set in the shell: the adapter refuses to start without an
explicit key, because the SDK's other credential sources fetch tokens outside the network
allowlist. The SDK's own retries are off, no fallback model is ever requested, and every
request goes only to the hosts in `allowed_hosts`. Spend is metered from `configs/prices.v1.json`
and `sweep` stops at 95 percent of `spend_cap_usd`. Run `probe` and `estimate` first and
read the forecast before any sweep.

The other transports are `bedrock` (API key only, model ids with the `anthropic.` prefix),
`vertex` (a static access token, which expires after about an hour) and `foundry`. Their
region, project or resource go in the model's `transport_options` so they are hashed with
the config.

## Verifying a published snapshot

Three tiers, from minutes to hours (`ARCHITECTURE.md` Section 14):

1. Re-analysis, minutes, any machine: `uv sync --locked` then
   `uv run islands verify results/s1`. Checks SHA256SUMS and the freeze lock, re-runs the
   statistics from the archived `runs.jsonl` into a temporary directory and compares every
   analysis file: identical bytes on the same stack, numbers within a relative 1e-9
   elsewhere. `islands analyze DIR --check` runs the comparison alone.
2. Re-grading: add `--regrade` to recover each accepted submission from its transcript,
   check it is the one runs.jsonl recorded, and score it again with the frozen grader.
3. Re-execution, hours, a GPU: add `--reexecute --cells 2:0.0 6:0.5` with the model's
   server running; each cell is run again beside the results and compared by success count
   and transcript by transcript. For hosted models this tier is a new dated measurement.
   `uv run islands replay RUN_ID --results DIR` re-executes one run and reports where its
   transcript first diverges, if it does.

`verify` exits 0 only when every check passes. Unfrozen results (the selftest, exploratory
runs) are verified with `--exploratory`, which skips the lock.

A year later, `git checkout v1.0.0` followed by the sequence above reproduces every
transcript bitwise, and every field of `runs.jsonl` except the wall-clock ones (`seconds`
and the request dates), for local models on equivalent hardware when the determinism gate
passes; and every statistic from the archived `runs.jsonl` on any machine.

## Layout

See `ARCHITECTURE.md` Section 4. In short: `src/islands_harness/` is the package,
`tasks/invoice_extraction/` the public task, `configs/` the snapshot and pre-registration,
`server/llamacpp/` the pinned server install and launch scripts, `models/` the weight locks
(weights themselves in the gitignored `models/weights/`), `schemas/` the
exported JSON Schemas, `tests/` the unit, property and simulation tests, `results/` the
gitignored outputs, and `docs/` the method note, client mode and reproducibility guides.

## Client mode

The same package pointed at a client's task, model and tools, run on their machine or
on-site; nothing is sent to the public program. `uv run islands client init DIR`,
`uv run islands client run DIR`, `uv run islands distance --tools K --fault F`. See
`docs/client-mode.md` and `ARCHITECTURE.md` Section 13.

## License

Apache-2.0.
