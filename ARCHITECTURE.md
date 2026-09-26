# Islands Harness: Architecture

Design version 1.0, dated 2026-09-25. It answers pre-registration draft r3 and the three judged proposals. Where it departs from r3, it says so, and the departure becomes a named field in r4.

## 1. Overview

The harness runs one agent through one fixed task thousands of times under controlled conditions and reports the shape of the region where it works. Two axes: tools offered (1 to 6) and injected tool-fault rate (0 to 50 percent in six rungs). Each cell gets 100 runs locally and 50 hosted. The outputs are a heatmap with intervals, five summary numbers, one pre-registered logistic fit, raw logs of every run, model identifiers, a dated one-line statement, and a static JSON the website renders.

The chosen design is the standalone Python proposal with grafts from the other two. The harness is one Python package, `islands_harness`, with one CLI, `islands`. It owns the agent loop, the tool boundary and the fault injector. Every stage reads and writes files with versioned schemas. The statistics are statsmodels, scipy and firthmodels calls, verified by simulation against known truths. Every random quantity is a hash of the config and an index. The pre-registration is data, hashed and checked before every analysis. Everything runs natively on Windows: the harness through uv, the local models on a pinned llama.cpp build with one server slot.

Grafted from the Inspect proposal: the statistics and verdict layer, sampling with seeds above temperature zero, the three-regime determinism gate, the two noise floor protocols, model lock files, the probe and the spend estimate. Grafted from the TypeScript proposal: hash-derived randomness, the runtime network allowlist, SDK retries off, the `executed` flag on faults, replay and calibrate commands, a pure SVG heatmap, and round-robin cell ordering.

## 2. Decisions

Each decision names the rejected alternative and the reason.

**D1. A standalone package that owns the loop.** The instrument is the boundary between the model and the tools, so the loop is one file of about 150 lines with one path from a tool call to a tool implementation. Rejected: Inspect AI's react() agent. Its defaults are hidden knobs: it appends its own prompt text, nudges the model when it stops without a tool call, executes parallel tool calls concurrently, truncates tool output, and returns its own text for malformed calls. Each changes turn counts or no-submission rates, and each release reopens the audit. Rejected: the TypeScript harness, whose hand-rolled statistics need a Python verifier anyway, so one maintainer keeps two toolchains.

**D2. Python 3.12 managed by uv.** uv installs the interpreter, resolves one lockfile for Windows and Linux, and enforces `--locked`. Rejected: Python 3.13 or 3.14, which add nothing the harness needs and one more place a client machine may lack wheels; pip-tools and conda, which do not install the interpreter.

**D3. Two provider adapters.** OpenAI-compatible endpoints are called through httpx with the Chat Completions wire format written by the harness, so every request body is logged verbatim and nothing is retried or reshaped by a library. Anthropic is called through the official `anthropic` SDK on the Messages API with `max_retries=0`, and through the same SDK's Bedrock, Vertex and Foundry clients. Rejected: the `openai` SDK, which retries silently by default and whose typed parser rejects small deviations from local servers; LiteLLM (hundreds of transitive packages in a lockfile that must hold for a year); Anthropic's OpenAI-compatible layer (a testing shim that drops features).

**D4. Sampling above zero with derived seeds.** Local models run at temperature 0.7 and top_p 1.0 with a per-run seed derived from the root seed. Hosted Claude models sample at the provider default, because current models reject temperature; the recorded configuration is effort plus thinking mode. Rejected: temperature 0, which on an invariant stack turns 100 runs into 100 greedy decodes of 100 documents, so the noise floor is zero by construction and the per-cell interval has no sampling interpretation.

**D5. Stateless randomness.** Every random draw is `unit(sha256(scope, root_seed, indices))` mapped to [0, 1). No `random.Random` object exists in the run path. Rejected: one seeded stream per run, which needs draw-count bookkeeping to stay aligned across cells.

**D6. A runtime network allowlist.** The provider layer builds its HTTP client on a transport that refuses any host not in `allowed_hosts`, and the harness owns and logs every retry. Rejected: a socket-blocking test alone, which proves the commit but not the client's machine.

**D7. Library statistics, verdicts as code, simulated truth.** Wilson intervals and the GLM come from statsmodels, Firth from firthmodels, curve fits from scipy. The verdict is a function with fixture tests, and `simulate.py` produces snapshots of known shape for coverage tests. Rejected: hand-rolled numerics, and notebooks, which cannot be pinned or asserted against.

**D8. Pre-registration as hashed data.** `configs/preregistration/r4.yaml` holds every decision rule as a named field. `islands freeze` hashes it with the prompt, tool schemas, dataset, grader, loop source and config, and `analyze` refuses to run if any hash differs. Rejected: rules as prose, which an auditor cannot check.

**D9. Everything native on Windows; local models on llama.cpp.** The local models run on llama.cpp's llama-server from build b11191, installed from the project's Windows CUDA release zips checked against their published SHA-256 (`server/llamacpp/install.ps1`), one model at a time, bound to 127.0.0.1. The author chose llama.cpp: it runs natively on Windows, has first-class GGUF weights for both models, and supports consumer Blackwell GPUs. Rejected: vLLM's batch-invariant mode in WSL2 and Docker, which documents determinism across batch composition but adds two layers one person must keep working, and whose MXFP4 kernels on the RTX 5080 were unverified; it remains the path for a rented-GPU track if one is added. Rejected: llama.cpp with several slots or prompt caching, both documented by the project as sources of nondeterminism. Deferred: llama.cpp's opt-in deterministic CUDA mode (pull request 16016), unmerged at the pinned build, which also targets BF16 and FP16 matrix paths rather than the quantized weights used here.

**D10. Two stronger local models that fit 16 GB, and a smaller third.** gpt-oss-20b in the MXFP4 format its weights were released in (12.1 GB, `ggml-org/gpt-oss-20b-GGUF`), and Qwen3-14B as Qwen's official Q5_K_M GGUF (10.5 GB, `Qwen/Qwen3-14B-GGUF`): two model families with serious tool-calling ability. Each leaves room for an f16 KV cache at 16k context with one slot. Added 2026-09-26: Qwen3-4B as Qwen's official Q5_K_M GGUF (2.9 GB, `Qwen/Qwen3-4B-GGUF`), a third replicate from the same family and quantization as Qwen3-14B, so the pair differs in size only; it was first rejected as too weak to transfer, and joined when calibration showed the task at the ceiling for the larger models. Rejected: Llama-3.2-3B, weaker at tool calling. Rejected: Qwen3-14B-AWQ, which llama.cpp cannot load, because AWQ is a vLLM and Transformers format. Rejected: Q8_0 (15.7 GB) and models of 27B and up, which do not fit beside a KV cache. Q6_K (12.1 GB) is allowed only if `doctor` finds the VRAM. Quantization is part of each model's identity; it is recorded, never converted.

**D11. Hosted headline on Anthropic with fallbacks off.** (Snapshot 1 is local only, by the author's decision of 2026-09-26; the hosted model's entry is kept in `configs/hosted-models.yaml` and this decision applies when a hosted model joins a later snapshot.) The default is `claude-opus-5` with adaptive thinking and effort `medium`. Server-side refusal fallbacks are never sent, because a fallback swaps the model under measurement. `tool_choice` is `auto`, never forced, and tools are sent without `strict`, so argument validation is the model's job on every provider. Rejected: thinking disabled, which on Opus 5 has a documented failure mode where tool calls appear as visible text.

**D12. One server slot, a three-regime gate, and a floor in the sweep's regime.** Each llama-server runs with one slot and prompt caching off, so batch composition cannot vary and every request is processed from scratch. The gate replays fixed runs serially, four at a time in shuffled order (they queue at the slot), and under filler load; the floor then runs in the sweep's regime. Rejected: several slots for throughput, which llama.cpp documents as nondeterministic. The cost is serial throughput: smoke runs on 2026-09-25 took 2 to 6 seconds each, so roughly 6 to 17 GPU hours per local model per snapshot.

**D13. Threshold by calibration, not by fiat.** The success threshold tau and the dataset difficulty are set in a discarded pilot so that the anchor cell lands in a pre-registered band. Rejected: tau = 1.0, a floor effect waiting to happen.

**D14. A fit ladder that ends in a named category.** Maximum likelihood first, Firth on separation, then a distinct `not_estimable` category with an instrument check. Rejected: declaring any saturated column not estimable; a 100 percent column among mixed cells does not separate the likelihood.

**D15. Client tool axis relative to the client's registry.** A client with twelve tools sees cells around twelve, and the distance report never extrapolates outside the sampled grid. Rejected: inheriting the public 1 to 6 axis, which never reaches the client's operating point.

**D16. Three small choices.** The heatmap is pure SVG with the site's CSS variables (rejected: a required matplotlib dependency). Timeout faults do not sleep, because turns are the recovery unit (rejected: real sleeps, which multiply hosted cost for nothing). Faults apply to every tool including `submit_record` (rejected: exempting the terminal tool, kept as a client option).

## 3. Components and data flow

The package is a pipeline around one mutable artifact, `runs.jsonl`. Everything upstream is a pure function of the config and the root seed; everything downstream is a pure function of `runs.jsonl` and the frozen pre-registration.

`config.py` loads the config and the pre-registration into frozen pydantic models with `extra="forbid"`, canonicalizes (UTF-8, LF, sorted keys) and hashes. `dataset.py` generates the synthetic invoices from a seed and a difficulty level, writes documents, gold records and per-file hashes, and verifies them. `specs.py` expands the sweep into cells and cells into run specs; run index i in every cell maps to the same document and sampling seed, so cells differ only in tool list and fault rate.

`tools.py` holds `ToolSpec` (name, description, JSON schema, async execute), the registry, the mixes, and `execute()`, the only function that turns a tool call into a result; `faults.py` wraps it. `providers/base.py` defines the protocol, the message types, a `Sampling` record, a capability table saying which fields each provider accepts, and the scripted fake provider. `netguard.py` builds the allowlisted transport and the retry policy. `openai_compat.py` and `anthropic_native.py` translate to the two wire formats.

`loop.py` runs one spec: system prompt, user prompt with the document id, then at most `max_turns` iterations of model call, sequential tool execution in emitted order, append. It ends at the first accepted submit, at a model stop without submit, at a limit, at a refusal, or at context overflow. A turn cut off at `max_tokens` ends the run as `limit_output` and none of its tool calls run, since their arguments may be partial; it scores as a failure like any other non-submission. It adds no text of its own. `runner.py` schedules specs through a semaphore in round-robin cell order, grades each finished run, appends one fsynced line to `runs.jsonl`, writes the transcript, meters spend, and resumes by run id. `stats/` reads `runs.jsonl` and writes `results.json`; `report.py` turns results into `snapshot.json`, `heatmap.svg`, `statement.txt`, `manifest.json` and the bundle; `provenance.py` holds the determinism gate, the jitter protocol, the probe and the spend estimate.

## 4. File layout

```
harness/
  pyproject.toml  uv.lock  .python-version  .gitattributes  LICENSE  CITATION.cff
  README.md                       install commands, quickstart, rerun recipe, Windows setup
  .github/workflows/ci.yml        windows-latest and ubuntu-latest: uv sync --locked, ruff, pytest, selftest
  configs/
    snapshot-1.yaml               the public snapshot, models inline
    snapshot-1.lock.json          written by islands freeze
    preregistration/r3.md         frozen prose as on the site
    preregistration/r4.yaml       decision rules as data, deposited with the DOI
    prices.v1.json                dated hosted price table
  tasks/invoice_extraction/
    system_prompt.md  tools.py  grader.py
    dataset/                      documents/, gold.jsonl, search_index.json, spreadsheets.json, LOCK.json
  models/                         gpt-oss-20b.lock.json, qwen3-14b.lock.json: repo, commit, size, sha256; weights/ gitignored
  server/llamacpp/install.ps1     pinned llama.cpp build; release zips checked by SHA-256
  server/llamacpp/serve.ps1       one model per server: one slot, no prompt cache, exact flags
  schemas/                        config.v1, runs.v1, snapshot.v1, exported by islands schema
  src/islands_harness/
    cli.py  config.py  rng.py  dataset.py  specs.py  loop.py  tools.py  faults.py  runner.py
    providers/  base.py  netguard.py  openai_compat.py  anthropic_native.py
    stats/  fit.py  recovery.py  verdict.py  simulate.py
    report.py  provenance.py
  tests/  test_harness.py  test_stats.py  fixtures/
  docs/  method-note.md  client-mode.md  reproducibility.md
  results/                        gitignored; one folder per snapshot and model
```

## 5. Config schema and example

One YAML file per snapshot describes what to run; the pre-registration file describes how to decide. Pydantic validates both, unknown keys are errors, and nothing outside these two files changes a measured quantity.

```yaml
schema_version: 1
snapshot: {id: s1, root_seed: 20261006, preregistration: configs/preregistration/r4.yaml, harness_tag: v1.0.0, outputs: results/s1}
task:
  dir: tasks/invoice_extraction
  system_prompt: system_prompt.md
  dataset: dataset
  grader: grader.py
  required_tools: [fetch_document, submit_record]
  mixes:
    A: [fetch_document, submit_record, search_web, calculator, read_spreadsheet, send_email]
    B: [fetch_document, submit_record, read_spreadsheet, search_web, calculator, send_email]
  rung1: [fetch_document]
agent:
  max_turns: 20
  max_tool_calls: 40
  token_limit: 60000
  time_limit_s: 600
  parallel_tool_calls: allow        # executed sequentially in emitted order
  tool_choice: auto
  strict_schemas: false
  on_stop_without_submit: end
  on_malformed_arguments: catalog_error
sweep:
  tools_axis: [1, 2, 3, 4, 5, 6]
  in_scope_tools: [2, 3, 4, 5, 6]
  fault_axis: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  mixes_to_run: [A]
  order: round_robin
  rung1_runs_per_cell: 20
faults:
  kinds: [timeout, garbled_payload, error_response, empty_result]
  weights: [1, 1, 1, 1]
  applies_to: all_tools
  nested_across_rates: true
  simulate_delay_s: 0
  garble: {char_fraction: 0.2, truncate: true}
probe: {runs_per_point: 20, tools_points: [2, 3, 4, 5, 6], fault_points: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
calibration:
  anchor_cell: {tools: 2, fault: 0.0}
  target_band: [0.70, 0.90]
  pilot_runs: 30
  difficulty_levels: [1, 2, 3]
  tau_candidates: [0.8, 0.9, 0.7]
  max_malformed_call_rate: 0.20
noise_floor:
  local: {cell: {tools: 3, fault: 0.2}, fixed_seed_reruns: 50, fixed_seed_documents: 20, varied_seed_reruns: 50}
  hosted: {cells: [{tools: 2, fault: 0.0}, {tools: 4, fault: 0.2}, {tools: 6, fault: 0.4}], reruns: 50, documents_per_rerun: 10}
determinism_gate: {specs: 20, replays_per_regime: 5, filler_requests: 200}
models:
  - id: gpt-oss-20b-local            # qwen3-14b-local has the same shape on port 8082
    provider: openai_compat
    base_url: http://127.0.0.1:8081/v1
    allowed_hosts: ["127.0.0.1:8081"]
    model: gpt-oss-20b
    lock: models/gpt-oss-20b.lock.json
    documents: 100
    epochs: 1
    concurrency: 1
    sampling: {temperature: 1.0, top_p: 1.0, max_tokens: 4096, seed: per_run}
    extra_body: {samplers: [top_k, top_p, min_p, temperature], top_k: 0, min_p: 0.0, reasoning_effort: medium, cache_prompt: false, parallel_tool_calls: true}
    expect: {server: llama.cpp, build: b11191, build_commit: 4b1a27fa0eb8..., weights_file: gpt-oss-20b-MXFP4.gguf, weights_sha256: 27cd6c43...5901, quantization: MXFP4, ctx_size: 16384, parallel_slots: 1, cache_prompt: false, flash_attn: "on", kv_cache_type: f16, batch_size: 2048, ubatch_size: 512, chat_format: GPT-OSS, deterministic: verify_with_gate}
  - id: opus5-hosted
    provider: anthropic
    transport: anthropic              # or bedrock, vertex, foundry
    model: claude-opus-5
    api_key_env: ANTHROPIC_API_KEY
    allowed_hosts: ["api.anthropic.com"]
    documents: 50
    epochs: 1
    concurrency: 4
    sampling: {temperature: null, thinking: adaptive, effort: medium, max_tokens: 4096}
    prompt_caching: true
    fallbacks: none
    prices: configs/prices.v1.json
    spend_cap_usd: 900
```

`configs/preregistration/r4.yaml` is the only other input the analysis reads. Its keys are `success` (rule, tau, band), `intervals` (wilson, newcombe, level), `p1` (model, scope, covariance, ladder, separation rule, curvature rule, peak range, drop row and SESOI, fault-cap check, bootstrap), `p2` (population, control, unit, rule, SESOI, secondary statistic), `below_resolution`, `rung1`, `noise_floor` (rerun counts, hosted block size, regime) and `descriptive`. Sections 9 to 11 give each value.

## 6. CLI

Every command is `uv run islands <cmd>`; `uv run` syncs the locked environment first.

- `doctor`: checks the toolchain, free VRAM against each weights file, weights hashes, API keys, each endpoint, the running llama-server's `/props` against the model's `expect` block, and every outbound host the config would contact.
- `dataset generate --seed --n 100 --difficulty 4`, `dataset verify`, `models fetch --verify`: build and check the dataset; download each local model at its pinned commit and verify every file's sha256.
- `serve start MODEL`, `serve stop`: run `server/llamacpp/serve.ps1` for one model with the pinned build and flags; refuse a weights hash mismatch. `smoke --model ID`: one run at rungs 2 and 6 at 0 and 50 percent faults.
- `calibrate --model ID`: the pilot at the anchor cell per difficulty level; prints the rate with its Wilson interval per candidate tau and the malformed-call rate.
- `freeze`: hash every fixed input, write the lock, refuse a dirty tree. From here every run command refuses a changed hash.
- `verify-determinism --model ID`: the three-regime gate; writes `determinism.json`.
- `probe`: 20 runs per probe point; writes `forecast.json` for deposit. `estimate --model ID --from forecast.json`: measured tokens times the price table times run counts.
- `noise-floor --model ID`: the fixed-seed and varied-seed protocols locally, the three-cell block hosted; writes `jitter.json`.
- `sweep --model ID --mix A [--resume] [--cells ...]`: appends to `runs.jsonl`, safe to kill, stops at the cap, never prints a verdict.
- `analyze RESULTS [--check]`: reads `runs.jsonl` and the hash-checked prereg, writes `results.json`; `--check` recomputes and diffs. `report RESULTS`: writes `snapshot.json`, `heatmap.svg`, `statement.txt`, `manifest.json`.
- `verify RESULTS [--regrade] [--reexecute --cells ...]`: the reviewer command; checks hashes, re-grades every transcript, re-analyzes, and optionally re-executes cells against the published jitter. `replay RUN_ID`: re-executes one run from its recorded spec and diffs the transcript.
- `selftest`: the whole pipeline on the fake provider. `bundle RESULTS`: the Zenodo bundle with `SHA256SUMS`. `schema`: the exported JSON Schemas.
- `client init DIR`, `client run DIR`, `distance --tools K --fault F`: client mode, Section 13.

## 7. Fault injection

Exactly one code path leads from a model's tool call to a tool implementation: `tools.execute()`, and `FaultInjector` wraps it. Nothing else can return a tool result and the loop has no retry logic, so a fault reaches the model unmodified and the model's response is the measurement. The model sees ordinary schemas and ordinary results.

Each tool call in a run gets a call index i, 0-based across all tools in execution order. Parallel calls in one assistant turn execute sequentially in emitted order, each with its own index, and all results go back in one message as both APIs require.

Decision: `u1 = unit(sha256("fault", root_seed, model_id, mix, doc_id, epoch, i))`; a fault fires when `u1 < rate`. The scope omits the tool count and the fault rate on purpose, so the schedule is nested: a call faulted at 10 percent is faulted at 20 through 50 percent, and at rate 0 nothing fires. These common random numbers lower the variance of every contrast and correlate cells, which the cluster-robust covariance by document absorbs. Kind: `u2 = unit(sha256("kind", same indices))` mapped through the cumulative weights; garble positions come from `sha256("garble", same indices, k)`.

Kinds, frozen in `faults.py` and hashed by freeze:

- `timeout`: not executed. Result: a JSON error naming the tool and a 30 second timeout.
- `error_response`: not executed. Result: a JSON error with status 503.
- `empty_result`: not executed. Result: the empty string.
- `garbled_payload`: executed. `char_fraction` of positions in the genuine result are replaced from a fixed alphabet, then the string is truncated at a hashed position in [0.3, 1) of its length. A property test asserts the output never parses as JSON and never equals the original.

On `submit_record`, the three unexecuted kinds store no record, so the model must resubmit; the garbled kind stores the record and garbles the acknowledgement.

Not faults: transport failures to the model endpoint (429, 5xx, connection errors, timeouts) are retried inside the provider layer with exponential backoff, at most six attempts, each logged as `transport_retry`. If exhausted, the run is `aborted_transport`, excluded from every denominator, counted in the manifest, and re-executed once. Malformed tool arguments get the frozen catalog error and the tag `model_error`. Every fault event logs turn, call index, tool, argument hash, kind and `executed`.

Recovery accounting happens in `stats/recovery.py`. For a faulted run that recovered, the control is the run with the same document, epoch and tool column at fault 0. On a verified invariant stack the two transcripts are byte-identical up to the first fault, so the difference in turns is a genuine paired difference; on hosted models the pairing holds in expectation only, and the report says so.

Rung 1 offers `fetch_document` only, so every rung 1 run scores 0 by construction; it runs at `rung1_runs_per_cell`, is excluded from every statistic, and reports whether the model attempted a submission in free text.

## 8. Determinism and jitter

The harness never assumes a stack is deterministic; it measures the residual before the sweep. At the harness level every draw is hash-derived, prompts are byte-stable with no dates or run ids in model-visible text, tools are pure and offline, and file hashing normalizes CRLF to LF so Windows and Linux checkouts hash identically.

Local stack. llama.cpp build b11191 (commit 4b1a27fa0eb8) from the project's CUDA 13.4 Windows release, one server slot (`-np 1`), prompt caching off (`--no-cache-prompt`, and `cache_prompt: false` on every request), fixed batch and micro-batch sizes, flash attention on, an f16 KV cache, every layer on the GPU, and a per-request seed from the run spec. Every sampler is set on every request (`samplers`, `temperature`, `top_p`, `top_k`, `min_p`), because llama.cpp's default chain would otherwise apply top-k 40 and min-p 0.05 silently. llama.cpp makes no determinism guarantee; its documentation names prompt caching and batch-size changes as sources of nondeterminism, and both are removed here. Any claim the gate grants is scoped to the same GPU model, driver, CUDA runtime, build and weights file, all in the manifest.

The concurrency regime objection. A floor measured in one regime says nothing about a sweep run in another. With one slot the sweep itself processes one request at a time, so the floor and the sweep share a regime by construction; what remains to test is whether queueing and submission order change anything. The design answers in two steps.

First, the gate. `verify-determinism` takes 20 fixed (document, seed) specs from the anchor cell and executes each five times in three regimes: serial; submitted four at a time in shuffled order, so they queue at the single slot; and the same with 200 unrelated filler requests in flight. It hashes the canonical transcript (assistant messages, tool calls, arguments, tool results) of all 300 executions, and passes when all 15 hashes agree for every spec. On failure the operator reruns the gate once with flash attention off (`--variant flash_attn_off`, a config change made before the freeze). If it still fails, the stack is labelled non-deterministic, the sweep and the floor still run in the same regime, and the published floor is "run-to-run jitter of a non-deterministic llama.cpp stack", a weaker claim stated as such.

Second, the floor runs in the sweep's regime. The fixed-seed protocol executes a 20-document block of the floor cell 50 times in the sweep's regime and round-robin order. With one slot it checks the gate at scale rather than measuring something new, which is why r4 shrinks it from 20,000 runs to 1,000. Per rerun r it computes p_r and reports the transcript identity rate (share of reruns whose block hashes all equal rerun 1), the outcome flip rate (share of documents whose outcome is not constant across reruns) with a Wilson interval, and the jitter SD, SD(p_r). On a verified stack these are 100 percent, 0 and 0. The varied-seed protocol runs the same cell 50 times with the seed scope offset by rerun index and reports SD(p_r) beside the binomial SD sqrt(p(1 - p)/n) and their ratio phi; because documents are fixed, phi above 1 points at the harness, not at sampling.

Hosted. Three pre-registered cells, each rerun 50 times on a fixed 10-document block at the hosted concurrency, 1,500 runs. The literal reading of r3 (50 reruns of the full cell) costs 7,500 runs, four times the sweep, so r4 records the block reading. OpenAI-compatible endpoints receive `seed` and the harness stores `system_fingerprint` per response. Anthropic offers no seed, so the harness records model id, effort, thinking mode, response id and request dates, and the measured jitter is the only determinism statement made.

The same pinned llama.cpp build is the development path for dry runs and the default for client mode. Jitter is reported beside the per-cell intervals, never inside them: sampling error is the criterion for edges, jitter is a statement about the instrument.

## 9. Statistics

All statistics are pure functions of `runs.jsonl` and `r4.yaml`, with alpha = 0.05 and every random procedure hash-seeded. Bootstrap resample indices are written to `bootstrap/resamples.json` so an independent implementation reproduces the intervals exactly.

The estimand. For model m and cell c, let pi_d be the probability that a run on document d succeeds, over the model's sampling distribution as served under the fixed sampling configuration. The estimand is pi_c = (1/D) sum over d of pi_d: the success rate on this fixed document set under this harness. Documents are enumerated, not sampled, so no claim is made about other documents. With one epoch the estimator is p = k/n with n = D; its variance is (1/n^2) sum over d of pi_d(1 - pi_d), at most p(1 - p)/n, so the Wilson interval is conservative for this estimand, and the varied-seed floor measures the actual ratio. For hosted models the estimand carries the served dates.

Per-cell interval. For k successes in n runs and z = 1.959964, the Wilson score interval has centre (p + z^2/2n)/(1 + z^2/n) and half-width z sqrt(p(1 - p)/n + z^2/4n^2)/(1 + z^2/n), from `statsmodels.stats.proportion.proportion_confint(method="wilson")`; about 9.7 points at n = 100 near p = 0.5.

Difference between two cells. The Newcombe hybrid score interval (1998, method 10) built from the two Wilson limits; a difference is claimed when it excludes 0. Non-overlap of the two Wilson intervals, the chart's display rule, is stricter.

Primary fit for P1, one model at a time, mix A, in-scope runs. Covariates tc = tools - 4, tc2 = tc^2, f = fault rate in [0, 0.5]. Model logit P(success) = b0 + b1 tc + b2 tc2 + b3 f, fitted at run level by `statsmodels GLM(Binomial)` with `cov_type="cluster"` on document id. Fitted peak t* = 4 - b1/(2 b2), defined only when b2 < 0. The decision interval for b2 is the cluster-robust Wald interval. The decision interval for t* is a cluster bootstrap percentile interval: 2,000 resamples of documents with replacement, each refitted through the same fit ladder, with resamples where b2 is not negative counted, not dropped (they enter as minus or plus infinity by the sign of b1; the limits are the 51st draw from each end). The decision interval for the drop is the Newcombe interval of Section 11, as `r4.yaml` states; the bootstrap drop interval is published beside it as a descriptive companion.

Fit ladder. If the MLE does not converge, or any |b_j| exceeds 15, the pre-registered fallback is Firth's penalized likelihood from `firthmodels`, with the same cluster bootstrap for every interval. If Firth also fails, which happens only when the in-scope outcome has no variation, P1 is `not_estimable`. A column at 0 or 100 percent does not by itself trigger the ladder; separation is a property of the fit diagnostics. Fault-cap consistency: b3 must be at most 0 or have an interval including 0; otherwise the verdict is blocked and a harness anomaly is reported.

Recovery for P2. Population: faulted runs in in-scope cells of mix A that recovered. Statistic R = mean of turns(faulted) - turns(control), control as in Section 7. Interval: cluster bootstrap over documents, 2,000 resamples, percentile. Seconds and breakdowns by fault kind and row are descriptive. A secondary statistic, extra turns minus fault count, measures recovery cost beyond one mechanical retry per fault, because any retry adds a turn and makes the literal P2 easy to satisfy. The survivorship condition is stated in the output.

Fault-axis shape, descriptive. Per in-scope column and pooled, the six cell rates are fitted by maximum binomial likelihood under LIN, p(f) = a + b f clipped to (0, 1), and LOG3, p(f) = a/(1 + exp(s (f - f0))), with bounded Nelder-Mead (`scipy.optimize.minimize`) from every start of a fixed grid. The LOG3 plateau a lies in [0, 1] and the midpoint f0 in [-0.5, 1.0]: unbounded, the optimizer drifted to plateaus in the thousands with midpoints far left of the data, where the logistic is an exponential decay that imitates a straight decline, and straight declines came out indistinguishable instead of slope. Report AIC_LIN - AIC_LOG3, f0 and s: cliff if the difference is at least 2, slope if at most minus 2, indistinguishable otherwise.

Viable fraction: the share of the 30 in-scope cells with p above 0.5, and the share whose Wilson lower bound is above 0.5. Boundary width: along the tool axis in each row, the rungs between the first cell below 90 percent and the first below 10 percent, claimed only where adjacent Newcombe intervals exclude 0. Sensitivity: a fractional logit on the continuous score, and the P1 conditions recomputed at tau plus and minus 0.1.

Verification of the instrument. `simulate.py` draws snapshots from known truths: a band with peak 3.5, a continent with b2 = 0, a flat surface below resolution, and document heterogeneity in pi_d. `test_stats.py` runs 500 snapshots each at n = 100 and n = 50 and asserts Wilson coverage in [93, 97] percent, peak-interval coverage at least 93 percent, and a false-survival rate under the continent at most 5 percent; power under the band is reported, not asserted. One pre-registered model per proposition; every other number is labelled descriptive.

## 10. Success threshold policy

The grader returns a continuous score in [0, 1] from weighted field checks: invoice_number 2, invoice_date 1, vendor_name 1, currency 1, total_amount 2, line_items 3 with per-item partial credit. Strings are case-folded and whitespace-normalized, dates parsed to ISO, amounts matched within 0.005. The rules are frozen in `grader.py` and hashed.

The floor effect arises when the anchor cell (2 tools, 0 faults) sits near 0, and the ceiling when it sits near 1; in either case no edge can appear. The policy makes the anchor land in the band [0.15, 0.95] before anything is frozen, then freezes everything. The band was [0.70, 0.90] until the anchor study of 2026-09-26 (`docs/method-note.md` Section 6): with the easy corner anywhere from 0.12 to 0.97 the verdicts kept their power, coverage and false-survival bars, and with it at 0.998 a real edge was detected 1.2 percent of the time, because a ceiling compresses the drop out of sight. The ceiling, not the level, is the hazard.

Procedure. `islands calibrate` runs the anchor cell 30 times per model per difficulty level, on pilot runs that never enter a snapshot. Tau 0.8 is tried first, because it means every high-weight field is right and at most one low-weight field is wrong. If the anchor is out of band, the difficulty level changes first, tau second. If no combination lands in the band, the model is not a replicate in snapshot 1 and the method note says why. A model whose malformed tool-call rate at the anchor exceeds 20 percent is likewise ineligible, because its map would measure a parser, not an agent. The chosen difficulty, tau and band are written into `r4.yaml` and the config, hashed by freeze, and deposited; after that nothing about the threshold changes.

Why this avoids the floor. The anchor is guaranteed mid-to-high, so the surface has room to fall along both axes. Cells at 0 percent far from the anchor are the cliff, not a defect, and the fit ladder handles the separation they cause.

## 11. Verdict policy

`verdict.py` applies `r4.yaml` mechanically and writes `verdict.json` with every condition, its value and its pass flag, plus `statement.txt`.

Instrument check first. If the anchor cell's observed rate falls outside the calibration band by more than its Wilson half-width, the snapshot is labelled `instrument_out_of_calibration`; descriptives publish, neither proposition moves, and the statement says so.

P1 conditions. C1 curvature: b2 below 0 with its decision interval excluding 0. C2 peak: the bootstrap interval for t* inside [2, 6]. C3 drop: in the 0 percent row, the Newcombe interval for p(peak column) - p(6 tools) excludes 0, where the peak column is round(t*) clipped to [2, 6]. P1 survives when all three hold. P1 is restricted when b2's interval lies entirely above 0, or t*'s interval lies entirely outside [2, 6], or the drop interval excludes 15 points on the small side. P1 is below resolution when b2's interval includes 0 and the drop interval includes both 0 and 15 points, the equivalence reading of "the intervals swallow every edge". P1 is blocked when the fault-cap check fails.

P1 not estimable. If the ladder reaches `not_estimable`, P1 receives that category, distinct from below resolution, and the statement names the reason (no variation in scope). The descriptive outputs and the Wilson map still publish.

P2 conditions. P2 survives when the lower bootstrap bound of R is above 0, is restricted when the interval includes 0 and its upper bound is below 1 turn, and is below resolution otherwise. The secondary net-of-retries statistic moves nothing in snapshot 1.

Statement format, one line per model and proposition: `2026-11-15 | s1 | gpt-oss-20b-local | mix A | P1: survives (peak 3.4 tools [2.9, 4.1]; drop 41 points [27, 53]) | P2: below resolution (+0.4 turns [-0.6, 1.3])`.

## 12. Outputs and the site JSON

Per snapshot and model under `results/s1/<model>/`: `runs.jsonl` (one RunRecord per run: cell, doc_id, epoch, seeds, outcome, score, success, turns, seconds, tokens, cost, fault events, first_fault_turn, recovered, response model id and fingerprint, transcript path), `transcripts/<run_id>.json`, `cells.csv`, `results.json` (cells with intervals, fit, peak, drop, ladder status, recovery, shape, jitter, verdicts), `bootstrap/resamples.json`, `jitter.json`, `determinism.json`, `forecast.json`, `spend.json`, `verdict.json`, `statement.txt`, `snapshot.json`, `heatmap.svg`, `manifest.json`, `bundle.zip`, `SHA256SUMS`.

`snapshot.json`, schema `snapshot.v1`, is what `../mock` reads:

- `schema_version`: 1.
- `snapshot`: `id`, `kind` (`snapshot`, `exploratory` or `selftest`), `label`, `date_start`, `date_end`, `preregistration` {`version`, `doi`, `hash`}, `harness` {`version`, `commit`, `config_hash`}, `bundle_doi`.
- `models[]`, one entry per model and mix: `id`, `role` (`local`, `hosted` or `synthetic`), `mix`, `identity` {`model`, `revision`, `weights_verified`, `server`, `build`, `gpu`, `response_models[]`, `fingerprints[]`, `sampling`}, `stack` {`deterministic_verified`, `label`}, `grid` {`tools_axis`, `fault_axis`, `cells[]` {`tools`, `fault`, `n`, `k`, `p`, `lo`, `hi`, `in_scope`, `mean_score`, `faults_per_run`}}, `jitter` {`identity_rate`, `flip_rate`, `flip_lo`, `flip_hi`, `jitter_sd`, `phi`, `regime`}, `fit` {`estimator`, `coefficients`, `intervals`, `intervals_source`, `peak`, `peak_interval`, `drop`, `drop_interval`, `status`}, `summary` {`viable_fraction`, `upper_tool_edge`, `boundary_width`, `recovery_time`, `fault_axis_shape`}, `verdicts` {`instrument`, `propositions_move`, `p1` {`verdict`, `moves`, `conditions[]`}, `p2` {`verdict`, `moves`, `conditions[]`}}, `statement`, `spend_usd`, `files` {`runs_jsonl`, `results`, `verdict`, `manifest`, `heatmap`}, relative to snapshot.json.
- `links`: `method_note`, `preregistration`.
- Conventions shared with `results.json` and `verdict.json`: an unbounded interval end is the string `"inf"` or `"-inf"`, a quantity that could not be computed is `null`, and a proposition whose `moves` is false did not move (the instrument check failed).

Rung 1 cells carry `in_scope: false` and render blank; every descriptive number carries `descriptive: true`.

## 13. Client mode

Client mode is the same package pointed at the client's material. `islands client init DIR` scaffolds a private config, `system_prompt.md`, `tools.py` with the two required stubs, `grader.py` with the [0, 1] contract or a frozen judge prompt, a `dataset/` folder, and a `.gitignore` that excludes results and data.

Three seams. Dataset: documents plus gold records in the client's record schema, locked by hash. Tools: the client implements `ToolSpec` for their real tools or deterministic stubs, and the injector wraps them unchanged. Grader: executable checks or a judge with a pinned model and hashed prompt, with tau fixed before any run through the same calibration. Tool axis: the sweep removes tools from the end of the client's declared production order, so the axis is a list of counts around their operating point, and cells at the operating point always exist.

The deliverable. `islands distance --tools K --fault F` reports the observed nearest cell first: the rungs in each direction to the nearest cell whose Wilson lower bound falls below the client's target. The fitted success at (K, F) with its interval and the gradient in each axis come second, labelled, and are refused outside the sampled grid. A quick 4 by 4 pass at 30 runs is labelled below resolution by construction; the full 6 by 6 at 50 is what a report rests on.

Perimeter. The only network destination is the `allowed_hosts` list, enforced at runtime; Bedrock, Vertex, Foundry and Azure endpoints are supported in snapshot 1. `--redact` replaces document text and tool payloads in transcripts with hashes and lengths. The client runs it themselves or the author runs it on-site; nothing is ingested by the public program. Apache-2.0.

## 14. Pinning and reproducibility

Code. `pyproject.toml` declares ranges, `uv.lock` is the pin, `.python-version` is 3.12, and `[tool.uv] required-version` pins the resolver. CI and `sweep` run with `UV_LOCKED=1`. A tagged release precedes snapshot 1 and Zenodo archives it with a version DOI; the pre-registration, the lock file and the forecast are a separate deposit dated before any full cell.

Server and weights. `server/llamacpp/install.ps1` pins llama.cpp build b11191 and the SHA-256 of each Windows release zip, and writes `build.json` beside the install; the manifest records the build, commit, CUDA runtime and driver. `models/*.lock.json` holds repo id, commit, size and sha256 per file; `models fetch --verify` recomputes every hash, `serve.ps1` checks it again before starting, and the sweep refuses to start on a mismatch. The bundle carries every URL and hash; the files themselves are too large to redeposit.

Hosted. Pinned model id, every distinct response model string and fingerprint with first and last seen dates, effort and thinking settings, and the dated price table. A hosted snapshot is re-runnable only while the provider serves that id, and the site says so.

Three verification tiers. Re-analysis in minutes on any machine: `uv sync --locked` then `islands verify`. Re-grading: `--regrade`. Re-execution in hours locally: `--reexecute`, read against the published jitter; for hosted models this tier is a new dated measurement. A year later, `git checkout v1.0.0` followed by the Section 6 commands reproduces every transcript bitwise, and every field of `runs.jsonl` except the wall-clock ones (`seconds` and the request dates), for local models on equivalent hardware when the gate passes; and every statistic from the archived `runs.jsonl` on any machine. The selftest shows the pattern: two runs agree in all 1,320 transcripts and differ only in `seconds`.

Machine setup. On the author's workstation the following are missing and are installed with these commands. uv: `winget install --id=astral-sh.uv -e`, then reopen the shell. Python: `uv python install 3.12`, then `uv sync --locked` in the clone. llama.cpp: `powershell -ExecutionPolicy Bypass -File server\llamacpp\install.ps1`. Weights: two `uv run hf download` commands with pinned revisions (README, step 4), about 22.6 GB in total. The NVIDIA driver (591.86, reporting CUDA 13.1, at the time of writing) should be current, because the CUDA 13.4 build relies on minor-version compatibility. Git long paths: `git config --global core.longpaths true`. No WSL2, Docker or Hugging Face token is needed.

## 15. Threats to validity of the instrument

1. llama.cpp makes no determinism guarantee. One slot and no prompt cache remove its documented sources of nondeterminism, and the gate measures the actual stack; a failure degrades the claim, stated as such, not the snapshot.
2. Local models, quantized ones especially, emit malformed tool calls. This is agent behaviour, but it can floor the map. The eligibility rule at calibration excludes models above 20 percent malformed at the anchor.
3. The grader is auditable but strict; normalization rules are frozen and published, and the continuous score is reported beside the cut.
4. Common random numbers correlate cells, and tool count confounds call count. Cluster-robust covariance and document bootstraps handle the first; faults per run are reported per cell for the second.
5. Hosted drift and constraints: fingerprint or served-model changes are logged per response and shown beside the map; no seed, no sampling parameters, adaptive thinking and possible refusals are recorded, and fallbacks are off.
6. The P2 test is easy to satisfy in turns. The net-of-retries secondary statistic exists for r4 to promote if the author agrees.
7. Spend. Tokens per run are a guess until the probe measures them; `estimate` gates every hosted phase and the runner stops at 95 percent of the cap.
8. Windows specifics: CRLF, code pages, long paths and antivirus latency on fsync. Hashing normalizes line endings, every file opens with explicit UTF-8, transcripts are sharded into short paths, and `smoke` measures fsync cost.
9. One maintainer, and disappearing providers and artifacts. A small package, a one-file loop, verdicts as code, dated snapshots, pinned release hashes and Zenodo bundles with logs keep the statistics reproducible after the author stops or the model is gone.
10. Serial local throughput. One slot makes a local snapshot roughly 6 to 17 GPU hours per model (measured per-run times, 2026-09-25); `smoke` measures seconds per run before the sweep is scheduled, and `sweep` resumes after any interruption.

## 16. Build plan

Effort is for one person coding with agents. Days are focused working days.

- M0 Skeleton and loop, 3 days. Repo, uv, lockfile, CI, config and hashing, rng, specs, loop, tools, faults, fake provider, netguard, openai_compat. Done when `islands selftest` passes and a llama.cpp dry run on gpt-oss-20b yields a transcript with an injected fault.
- M1 Task and grader, 2 days. Generator with a difficulty knob, 100 documents, gold, lock, six tools, grader. Done when gold scores 1.0 and hand-built failure cases score as expected.
- M2 Runner, spend, Anthropic adapter, 3 days. Resume, fsync, round-robin, ledger, estimate, `anthropic_native` with Bedrock and Vertex. Done when kill-and-resume leaves no duplicate or gap and a 20-run hosted dry run stays under a 5 dollar cap. Status 2026-09-25: built; kill-and-resume verified live on gpt-oss-20b (300 planned, 14 before the kill, 286 after, none duplicated or missing); the hosted dry run waits for the owner's key and approval. The Bedrock transport takes an API key only (SigV4 would fetch credentials outside the allowlist) and Vertex takes a static access token.
- M3 Statistics and verdict, 4 days. `fit.py`, `recovery.py`, `verdict.py`, `simulate.py`, coverage and fixture tests. Done when coverage and false-survival numbers are in the method note. Status 2026-09-26: done. On 500 simulated snapshots per truth and size, Wilson coverage is 96.1 to 96.4 percent, peak coverage 93.4 and 93.8 percent, false survival at most 0.2 percent (`docs/method-note.md` Section 6). Eight rule completions found during the build, and fixes for five defects an independent review found, are in; the completions are in r4, marked CONFIRM BEFORE FREEZE.
- M4 Outputs and verification, 2 days. `report.py`, `analyze --check`, `verify`, `replay`, schema export. Done when the mock site renders a `snapshot.json` produced from selftest. Status 2026-09-26: done. The selftest runs a synthetic agent (`providers/synthetic.py`) through the full sweep on 40 documents, analyzes, reports and bundles it, then re-analyzes (byte-identical), re-grades every run, replays four runs (identical transcripts) and checks SHA256SUMS. `mock/map.html?snapshot=data/selftest-snapshot.json` renders it; the page without the parameter is unchanged.
- M5 Local stack, 2 days plus download and GPU time. llama.cpp install and serve scripts, model locks, doctor with `/props` checks, the three-regime gate on both local models. Done when the gate passes on both, or the failure is documented with its cause and the stack labelled non-deterministic. Status 2026-09-26: done. Doctor checks `/props` and the server process's command line against each `expect` block. The gate passed on both models with flash attention on: 300 executions each, every run's 15 transcripts identical (gpt-oss-20b in 18 minutes, Qwen3-14B in 22). Both stacks are labelled `deterministic_verified` (`docs/method-note.md` Section 7).
- M6 Calibrate, freeze, publish, 3 days. Calibrate every model, choose difficulty and tau, write `r4.yaml`, docs, tag v1.0.0, public repository, freeze, probe, deposit. Done when the repository is public and the deposit is dated before any full cell. Status 2026-09-26: `calibrate`, `calibrate --recommend`, `freeze --dry-run`, a freeze that refuses while anything is marked for confirmation, and `deposit` are built; the pre-registration hash now leaves out the deposit metadata (`frozen_on`, `deposit_doi`). The pilot found every model at 100 percent on levels 1 to 3. In response: the task gained level 4 (amounts must be computed), an anchor study (`docs/method-note.md` Section 6) widened the band to [0.15, 0.95] after showing the verdicts hold for anchors from 0.12 to 0.97 and fail at 0.998, and Qwen3-4B joined as a third replicate (gate passed). Difficulty 4 at tau 0.8 makes all three local models replicates (pilot 26, 12 and 6 of 30); the snapshot dataset was regenerated at level 4. The freeze waits for the author's confirmations; the public repository and the Zenodo upload wait for the author.
- M7 Noise floor, 1 day of attention and 1 to 2 GPU days. Done when `jitter.json` is on the site.
- M8 Snapshot 1, 2 days of attention and about a week of wall clock. Sweeps, analyze, verify, report, bundle, Zenodo, site update. Done when the statement line is published with its DOI.
- M9 Client mode, 2 days. `client init`, `client run`, `distance`, redaction, docs, a worked example with stubs. Done when a stranger can run the example from the README.

Total about 25 working days to snapshot 1. The freeze at the end of M6 is the point of no return.
