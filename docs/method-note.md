# Method note, snapshot 1 (draft)

Status, 2026-09-26: the statistics are built and verified against simulated truths. Nothing
has been frozen or measured on a real model. Sections marked *pending* fill in at
calibration (M5) and after the sweep.

This note explains how the harness turns runs into numbers and verdicts, and how well that
machinery performs when the right answer is known. The rules themselves live in
`configs/preregistration/r4.yaml`; `PREREGISTRATION.md` states them in prose. Where this
note and r4 disagree, r4 wins.

## 1. What is estimated

For one model and one cell (a tool count and a fault rate), the estimand is the success
rate on this fixed set of documents under this harness: the average over documents of each
document's probability of success. The documents are enumerated, not sampled, so no claim
reaches beyond them. A run succeeds when its grader score is at least tau.

## 2. Intervals

Each cell carries a 95 percent Wilson interval. Differences between two cells use the
Newcombe hybrid score interval, built from the two Wilson intervals. The chart draws
non-overlap of Wilson intervals, which is stricter than the Newcombe criterion.

With one run per document, the true sampling variance of a cell rate is at most the
binomial variance, and smaller when documents differ in difficulty. The Wilson interval is
therefore conservative for this estimand, which the simulations below confirm.

## 3. Proposition 1: an upper tool edge

One logistic regression per model on the in-scope runs (2 to 6 tools, every fault rate):
the log-odds of success are a quadratic in tool count plus a linear term in fault rate. The
covariance is clustered by document. The fitted peak is 4 - b1 / (2 b2) when the curvature
b2 is negative.

- **C1, curvature.** b2 is negative and its interval excludes 0.
- **C2, peak.** The bootstrap interval for the peak lies inside 2 to 6 tools.
- **C3, drop.** At zero faults, the success rate at the peak column is significantly above
  the rate at 6 tools.

The proposition survives when all three hold. It is restricted when the data rule out an
edge of useful size, and below resolution when they cannot tell. It is not estimable when
the fit cannot run, and blocked when faults appear to help, which would signal a harness
fault rather than a finding.

The fit starts with maximum likelihood. If that fails to converge or any coefficient
exceeds 15 in absolute value, the fit moves to Firth's penalized likelihood, with every
decision interval taken from the bootstrap. The bootstrap resamples documents 2,000 times,
refits each time, and writes every resample index to `bootstrap/resamples.json`. Any single
index can be recomputed from its coordinates.

## 4. Proposition 2: recovery costs time

Each faulted run that recovered is paired with the run on the same document, epoch and tool
column at zero faults. The statistic is the mean number of extra turns, with a document
bootstrap interval. It survives when the lower bound is above zero.

This conditions on recovery. A fault that ends a run without a submission contributes no
pair, so the statistic is the cost of recovering given that recovery happened. The cell
success rates carry the other half, and every output counts the faulted runs that did not
recover.

## 5. Rules completed during the build

The rules as first written left eight cases unstated. Each completion below is written into
r4 and marked to be confirmed before the freeze.

- **No peak in a resample.** A resample whose fitted curvature is not negative has no peak.
  It enters the peak percentiles as minus infinity when success falls with tools and plus
  infinity when it rises. The interval's limits are the 51st of 2,000 draws from each end,
  so an end opens when more than 2.5 percent of resamples lack a peak on that side, and C2
  then fails.
- **Refitting resamples.** Each resample goes through the same ladder as the primary fit:
  maximum likelihood, then Firth when that fails the separation rule. A resample whose Firth
  refit also fails is treated as having no peak, and counted. Without the ladder, a steep
  but genuine fault effect near the coefficient cap of 15 could open the peak interval.
- **No peak in the primary fit.** The drop is then measured from 2 tools to 6 tools. C3 fails
  by definition, but a precisely measured flat curve can still be restricted. Without this
  rule, most flat truths ended in an unnamed branch.
- **Peak column ties.** The peak column is t* rounded half up, so every implementation agrees
  at exactly 2.5 or 4.5 tools.
- **Order of categories.** The first category whose rule holds wins, in the order not
  estimable, blocked, survives, restricted, below resolution. One consequence needs a
  decision: when all three conditions hold but the whole drop interval lies below the
  15-point SESOI, the proposition survives rather than being restricted.
- **Unnamed branch.** Some cases match no named category: a fitted peak at 5.5 tools or
  beyond, where the drop column is 6 itself, a decline without an interior peak, or clear
  curvature with an unresolved drop. They are labelled below resolution and carry a
  `residual_branch` flag.
- **No recovery pairs.** Proposition 2 is not estimable when no recovered run has a control.
- **Recovery faster than control.** A recovery interval lying entirely below zero is
  restricted: it rules out a cost of one turn or more.

Three further guards need no confirmation because they only refuse to answer. A partial
sweep whose cells cannot identify the four coefficients, or a sweep of one document, is not
estimable. A pre-registration file that does not name the resampling rules and the order of
categories above is refused. When the calibration check fails, every stored verdict is
marked as not moving, not only the statement line.

The fault-axis shape fit, a descriptive statistic, bounds the logistic curve's plateau to
[0, 1] and its midpoint to [-0.5, 1.0]. Without bounds the optimizer reached plateaus in the
thousands with midpoints far left of the data, where the "logistic" is really an exponential
decay. With its extra parameter that decay imitated a straight decline, so straight declines
were labelled indistinguishable rather than slope: on 120 simulated columns, 56 moved from
indistinguishable to slope once the bounds were in place, and no label moved to or from
cliff. The bounded fit runs from every start of the same grid.

An independent read-only review of the statistics code, before the numbers below were
produced, found five defects: mixes pooled in the cell table, the coefficient cap misapplied
to resamples, uneven percentile tails, crashes on edge-case inputs, and stored verdicts that
ignored a failed calibration check. All five are fixed and covered by tests.

## 6. Verification against known truths

`simulate.py` draws complete synthetic snapshots from three truths, each with document
difficulty varying on the log-odds scale (standard deviation 0.8):

- **band**: a real edge, peak at 3.5 tools, success falling steeply by 6 tools;
- **continent**: no curvature, success flat across tools and falling with faults;
- **flat**: no curvature and a weak fault effect, success near 60 percent everywhere.

Each truth ran 500 times at 100 documents and at 50 documents, through the full pipeline:
fit, 2,000-draw bootstrap and verdict.

| Truth | Documents | Wilson coverage | Peak coverage | False survival | Power | Residual branch | Verdicts, of 500 |
|---|---|---|---|---|---|---|---|
| band | 100 | 96.2% | 93.4% | | 100% | 0% | 500 survive |
| band | 50 | 96.1% | 93.8% | | 100% | 0% | 500 survive |
| continent | 100 | 96.1% | | 0.0% | | 11.4% | 340 restricted, 160 below resolution |
| continent | 50 | 96.3% | | 0.0% | | 13.8% | 199 restricted, 301 below resolution |
| flat | 100 | 96.1% | | 0.2% | | 9.8% | 283 restricted, 216 below resolution, 1 survives |
| flat | 50 | 96.4% | | 0.0% | | 10.0% | 130 restricted, 369 below resolution, 1 blocked |

Run on 2026-09-26. No snapshot used the Firth rung and no bootstrap refit failed its rung.

The acceptance bars, set in `ARCHITECTURE.md` Section 9 before any simulation ran, are
Wilson coverage between 93 and 97 percent, peak coverage of at least 93 percent, and false
survival of at most 5 percent. Every row meets them.

- **Wilson coverage is about 96 percent,** above the nominal 95, as expected for a
  conservative interval under document heterogeneity.
- **Peak coverage is 93.4 and 93.8 percent,** below the nominal 95. The percentile bootstrap
  of a ratio of coefficients runs slightly narrow. The bar is met, but with little margin.
  In practice C2 is slightly easier to pass than its nominal level implies.
- **False survival is at most 0.2 percent,** far below the 5 percent bar, because survival
  needs three conditions at once.
- **Power under the band is 100 percent.** The band is a strong effect, so this says nothing
  yet about weaker edges.
- **The residual branch labels 10 to 14 percent of flat-truth snapshots.** It never fires
  under the band. That share is why r4 now names the branch instead of leaving it implicit.
- **Fewer documents shift flat truths from restricted toward below resolution,** as they
  should: at 50 documents the drop interval is wider, so a flat curve is ruled out less
  often.
- **The fault-cap check** blocked 1 of 1,000 flat-truth snapshots by chance.

Limits of this verification. None of the simulated snapshots triggered the Firth rung, so
its coverage is untested; the ladder's switching is covered by a unit test on separated
data. The truths share one document-heterogeneity level and one design size per row.
Weaker bands, heavier heterogeneity and designs with empty cells are not yet simulated.

To reproduce the table:

    uv run python -m islands_harness.stats.simulate

The assertions run on 40 snapshots per truth in the default test suite, and on 500 with
`uv run pytest -m full_simulation`.

### The anchor study, 2026-09-26

The calibration band says where the anchor cell (2 tools, no faults) must sit before the
freeze. The rule as first written asked for 70 to 90 percent, to keep away from a floor or
a ceiling where no edge can show. When the pilot found every model at 100 percent, the band
was tested instead of assumed: the same two shapes, a real edge and no edge, were simulated
with the easy corner moved from near zero to the ceiling, 500 snapshots each at 100
documents.

| Truth | Anchor rate | Power | Peak coverage | False survival | Verdicts, of 500 |
|---|---|---|---|---|---|
| edge | 0.12 | 100% | 94.4% | | 500 survive |
| edge | 0.26 | 100% | 93.2% | | 500 survive |
| edge | 0.45 | 100% | 94.8% | | 500 survive |
| edge | 0.97 | 100% | 93.4% | | 500 survive |
| edge | 0.998 | 1.2% | 89.6% | | 494 restricted, 6 survive |
| no edge | 0.19 | | | 0.0% | 332 restricted, 168 below resolution |
| no edge | 0.40 | | | 0.0% | 258 restricted, 242 below resolution |
| no edge | 0.97 | | | 0.0% | 475 restricted, 25 below resolution |

The verdicts hold with the easy corner anywhere from 0.12 to 0.97; the existing band truth
of Section 6 already had it at 0.58. They fail at 0.998: when the easy corner is perfect, a
real edge becomes almost invisible (6 tools at 0.983 against a peak of 1.000), and the
instrument wrongly restricts the proposition 494 times in 500. The ceiling, not the level,
is the hazard. r4 therefore widens the band to [0.15, 0.95], inside the tested range, and
does not accept a perfect anchor. Reproduce the table with
`uv run python -m islands_harness.stats.simulate --anchor`.

## 7. Calibration, jitter and results

*Pending:* the snapshot results.

### Calibration, 2026-09-26

The pilot ran the anchor cell (2 tools, no faults) 30 times per model at each difficulty
level, on 30 pilot invoices per level generated from their own seeds, so no pilot document
is a snapshot document. Anchor successes at tau 0.8:

| Model | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| gpt-oss-20b | 30 of 30 | 30 of 30 | 30 of 30 | 26 of 30 |
| Qwen3-14B | 30 of 30 | 30 of 30 | 30 of 30 | 12 of 30 |
| Qwen3-4B | 30 of 30 | 30 of 30 | 30 of 30 | 6 of 30 |

On levels 1 to 3 every model was at the ceiling: 269 of 270 runs scored exactly 1.0, the
exception losing one line item. An exploratory probe outside any snapshot, level-3 invoices
three times longer, also scored 30 of 30 for gpt-oss-20b and Qwen3-14B, so length is not
what makes the task hard for these models.

Level 4 was added in response: everything in level 3, but no line amount, subtotal, tax
amount or total is printed, and the document states how to compute them. A careful reading
still scores 1.0, which the task tests check on every level-4 document. At level 4 the
models separate. gpt-oss-20b, which reasons before answering, missed the total in 10 of 30
runs. Qwen3-14B and Qwen3-4B, run in their non-thinking mode, missed the total in every
run; their totals match no systematic misreading (not the subtotal, not a skipped or
misapplied tax), so these are arithmetic slips, not an ambiguous rule. From 4 tools up
the calculator, and from 5 tools the spreadsheet with each line's amount, can do what the
reader otherwise must, so at this level more tools can help before they hurt.

Choice. With the band [0.15, 0.95] (Section 6), difficulty 4 and tau 0.8 make all three
local models replicates, at 87, 40 and 20 percent; no other combination makes more than
one. Qwen3-4B sits at the band's lower edge (6 of 30), so the instrument check on the full
100 documents decides whether it stays a replicate. The determinism gates ran on the
level-2 documents; the gate certifies the stack (GPU, driver, build, weights, flags), and
the fixed-seed noise floor rechecks it on the snapshot's own documents at scale.

### The determinism gate, 2026-09-26

Before the gate ran, `islands doctor` confirmed each server against the config. `GET /props`
matched the build, the single slot, the 16k context, the weights file, the quantization and
the chat template. The server process's own command line matched every flag: prompt cache
off, flash attention on, f16 KV cache, batch 2048 and micro-batch 512. Both weights files
matched their locks.

The gate then replayed 20 fixed runs of the anchor cell five times in each of three regimes:
one at a time; four at a time in shuffled order, queueing at the single slot; and the same
with 200 unrelated filler requests of 5 to 400 words in the queue. The canonical transcript
hash covers every model message, tool call, argument, tool result and, for gpt-oss-20b, the
reasoning text.

| Model | Executions | Specs with one transcript | Filler failures | Minutes | Label |
|---|---|---|---|---|---|
| gpt-oss-20b, MXFP4 | 300 | 20 of 20 | 0 | 18 | deterministic_verified |
| Qwen3-14B, Q5_K_M | 300 | 20 of 20 | 0 | 22 | deterministic_verified |
| Qwen3-4B, Q5_K_M | 300 | 20 of 20 | 0 | 9 | deterministic_verified |

Every one of the 15 executions of each run produced the same transcript, byte for byte,
whatever ran before it in the queue. The fallback of rerunning with flash attention off was
not needed.

Scope of the claim. It holds for this GPU (RTX 5080, driver 591.86), llama.cpp build
b11191 with CUDA 13.4, these weights files and these server flags, all recorded in each
model's `determinism.json`. The gate's runs are short: anchor-cell runs of two or three
turns without faults. Longer transcripts, with more tools, injected faults and more turns,
are checked at scale by the fixed-seed noise floor, which reruns a 20-document block of the
3-tool, 20 percent cell 50 times in the sweep's own regime.

### The noise floor

`islands noise-floor` runs the two local protocols of r4 on the floor cell (3 tools, 20
percent faults, mix A) in the sweep's own regime: one request at a time on the single
llama.cpp slot, in round-robin order, through the same runner, loop, fault injector and
grader as the sweep. Its runs are stored under `results/s1/<model>/floor/`, apart from the
sweep's, and it resumes run by run.

- Fixed seed. The first 20 indices of the cell (20 documents, each with its sampling seed
  and its fault draws) are executed 50 times as identical specs. On a stack the gate
  verified, nothing but the stack can make two reruns differ.
- Varied seed. The whole cell, 100 documents, is executed 50 times with the sampling
  seed's scope offset by the rerun index. Documents and fault draws stay fixed, so only
  sampling varies between reruns.

Conventions, committed before the first floor run (commit ccbcd68):

- The identity rate is the share of reruns 2 to 50 whose 20 canonical transcript hashes
  all equal rerun 1's, document by document. Rerun 1 is the reference and is not counted.
- The flip rate is the share of the 20 block documents whose success (score at least tau)
  is not the same in all 50 reruns, with its Wilson 95 percent interval.
- The jitter SD is the standard deviation, with the n - 1 denominator, of the 50
  fixed-seed rerun success rates p_r.
- For the varied-seed protocol, SD(p_r) is reported beside the binomial SD
  sqrt(p (1 - p) / n) at the mean rate p, with n = 100 runs per rerun, and phi is their
  ratio. Because the documents are fixed, phi above 1 points at the harness, not at
  sampling.
- A run that ends aborted_transport twice is counted and left out of every rate.

On a verified stack the fixed-seed statistics are 100 percent, 0 and 0. Each invocation
appends a session (UTC start and end, harness commit, server evidence, GPU) to
`floor/sessions.jsonl`, and `jitter.json` repeats them, so the record shows that the floor
ran after the deposit (published 2026-09-26 20:24 UTC) and on which code.
`scripts/noise_floor_all.py` runs the three models one after another, fastest first.

*Results.* All 18,000 floor runs ran after the deposit: the first at 2026-09-26 21:00:50 UTC,
the last at 2026-09-27 21:54:41 UTC, on harness commits recorded per session.

Fixed seed, 50 reruns of a 20-document block:

| Model | Identity rate | Documents that flip (Wilson 95%) | Jitter SD | Block rate, every rerun |
|---|---|---|---|---|
| Qwen3-4B | 100% | 0 of 20 (0 to 0.16) | 0 | 0.40 |
| Qwen3-14B | 100% | 0 of 20 (0 to 0.16) | 0 | 0.20 |
| gpt-oss-20b | 100% | 0 of 20 (0 to 0.16) | 0 | 0.80 |

Every fixed-seed run of every model reproduced its rerun-1 transcript byte for byte, on
runs with three tools and injected faults. Qwen3-14B's block spanned three server launches
and a hibernation (below), so the claim also covers a restarted server.

Varied seed, 50 reruns of the whole cell, 100 documents each:

| Model | Mean rate | Range of p_r | SD(p_r) | Binomial SD | phi | Documents that flip | Aborted |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | 0.240 | 0.23 to 0.27 | 0.0084 | 0.0427 | 0.20 | 4 of 100 | 0 |
| Qwen3-14B | 0.311 | 0.30 to 0.33 | 0.0074 | 0.0463 | 0.16 | 7 of 100 | 0 |
| gpt-oss-20b | 0.782 | 0.71 to 0.84 | 0.0286 | 0.0413 | 0.69 | 70 of 100 | 5 |

Reading. phi is below 1 for every model: a cell's Wilson interval is wider than the spread
that new seeds produce on these documents, so the intervals are conservative for this
estimand (Section 1), and nothing in the harness adds variance. The two Qwen models are
close to deterministic per document at their vendor sampling (4 and 7 of 100 documents
change outcome under another seed); gpt-oss-20b at temperature 1.0 is not (70 of 100),
which is why its phi is the highest. None of these numbers enters an interval or a verdict.

Two incidents, both on record:

- Hibernation. On 2026-09-27 at 00:49 UTC Windows hibernated, chosen from the Start menu,
  while Qwen3-14B's fixed-seed block was running (714 runs done), and woke at 02:06. The
  request in flight hung, so the floor was stopped at 02:11 and resumed on a restarted
  server, then stopped once more at 02:15 to switch to the driver with a stall watchdog
  (commit 57f1845). The two killed sessions appear in `floor/sessions.jsonl` as entries
  reconstructed from the driver log. No run was lost or duplicated.
- Server parse failures. 5 of gpt-oss-20b's 5,000 varied-seed runs ended aborted_transport.
  The server log gives the cause: with those seeds the model's first reply did not match
  llama.cpp's gpt-oss output grammar, and the server answered HTTP 500 ("The model produced
  output that does not match the expected peg-native format") on all 12 attempts. They are
  model failures that the transport layer reports as aborts; under the frozen rule they
  leave the rates (0.1 percent of the protocol's runs). Neither Qwen server logged an error.
  The same can happen in the sweep, where the frozen rule would drop a model failure from
  the denominator; how the sweep records and reports these runs is settled before M8.

### Decided before the first sweep run, 2026-09-27

The author decided the following on 2026-09-27, after the noise floor and before any sweep
run (commit 294833f and this note):

- Every run that ends aborted_transport records why, as an `abort_cause` event in
  runs.jsonl: `unparseable_model_output` when the server answered that the model's own
  reply did not match its chat format (llama.cpp: "does not match the expected ...
  format"), `transport` otherwise. Nothing else about any run changes.
- The verdicts keep the frozen rule of r4: every aborted_transport run leaves every
  denominator.
- Beside the verdicts, labelled descriptive, the snapshot reports the number of each kind
  of abort per cell, and every cell rate and the P1 and P2 statistics recomputed with the
  `unparseable_model_output` runs counted as failures (score 0). Where a verdict would
  read differently under that count, the report says so next to the verdict.
- `islands sweep` refuses a local server that differs from the config, as the gate and
  the floor do, and appends each session (UTC times, harness commit, server evidence,
  GPU) to `sessions.jsonl` beside runs.jsonl.
- Both tool mixes run; only mix A moves the verdicts, and mix B is reported as
  descriptive (decided 2026-09-26). `scripts/sweep_all.py` runs the three models one
  after another, fastest first, mix A before mix B.
