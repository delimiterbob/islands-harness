# Pre-registration for snapshot 1 (r4)

Status: CONFIRMED by the author on 2026-09-26, decision by decision, and frozen by
`islands freeze`. Nothing in snapshot 1 has been measured. This document mirrors the
decision rules of draft r3 (the site's Map page, 25 September 2026) as sharpened in r4.
The machine-readable copy is
`configs/preregistration/r4.yaml`; the analysis reads only that file, and where the two
disagree the YAML is what was run and the disagreement is an error in the method note.

| Field | Value |
| --- | --- |
| Version | r4 (supersedes r3, the prose on the site) |
| Frozen on | 2026-09-26 (lock at commit 79b627f97627) |
| Deposit DOI | [10.5281/zenodo.22982434](https://doi.org/10.5281/zenodo.22982434) (Zenodo, published 2026-09-26, before any snapshot run; the deposited copy of this file predates the DOI and reads "pending") |
| Hash of `r4.yaml` | d52f100b9a77f4c9edf29b3147e2edcb8b63845723377159c79c837c56bbdb12 (rules only; `frozen_on` and `deposit_doi` excluded) |
| Harness tag | v1.0.0 (local; the public repository follows the deposit, and the lock ties the code to it) |
| Config hash | 6471cd3a4bb2924a02d877c09bb26305e326dc78230fc7345183ecd562f6924c (`configs/snapshot-1.lock.json`) |
| Author | Robert Encarnação |

## 1. Propositions

**P1, upper tool edge.** Success is a concave function of tool count with an interior peak,
and the drop from the peak to six tools exceeds sampling error.

**P2, recovery cost.** Recovery after an injected fault costs measurable time against
unperturbed controls.

Both propositions are tested by one pre-registered statistic each. Every other number in
the snapshot is descriptive and labelled so.

## 2. Fixed inputs

- Task: synthetic invoice-style extraction, `tasks/invoice_extraction`. Documents:
  N = 100, difficulty level 4 (chosen by calibration; seed 20261006).
- Agent configuration: `configs/snapshot-1.yaml`, hashed. System prompt hashed byte for byte.
- Grader: `tasks/invoice_extraction/grader.py`, continuous score in [0, 1], weights
  invoice_number 2, invoice_date 1, vendor_name 1, currency 1, total_amount 2, line_items 3.
- Models: three local replicates, confirmed 2026-09-26. Local models run on
  llama.cpp build b11191 (commit 4b1a27fa0eb8), one server slot, prompt caching off:
  `gpt-oss-20b-local` (`ggml-org/gpt-oss-20b-GGUF`, `gpt-oss-20b-MXFP4.gguf`, sha256
  27cd6c43...5901, reasoning effort medium, temperature 1.0, top_p 1.0), and
  `qwen3-14b-local` (`Qwen/Qwen3-14B-GGUF`, `Qwen3-14B-Q5_K_M.gguf`, sha256 e7c9aba1...3e31,
  non-thinking mode, temperature 0.7, top_p 0.8, top_k 20). Full hashes and repository
  commits are in `models/*.lock.json`. Each is a replicate only if the gate passes or the
  stack is labelled non-deterministic, and calibration lands in band. Added 2026-09-26:
  `qwen3-4b-local` (`Qwen/Qwen3-4B-GGUF`, `Qwen3-4B-Q5_K_M.gguf`, sha256 aca59686...6533f),
  sampled like Qwen3-14B. All three local stacks passed the determinism gate on 2026-09-26.
  Hosted: none in snapshot 1 (author's decision, 2026-09-26); the hosted rules below apply
  only if a hosted model joins a later snapshot.
- Tool mix A: fetch_document, submit_record, search_web, calculator, read_spreadsheet,
  send_email. Mix B swaps read_spreadsheet to third. Both mixes run in snapshot 1 (author's
  decision, 2026-09-26). Only mix A moves the propositions, as r4 scopes them; mix B's map
  and statistics are reported as descriptive. At level 4 the spreadsheet holds each line's
  amount, so mix B brings a helpful tool in at 3 tools instead of 5.
- Sweep: tools 1 to 6 by fault rate 0, 10, 20, 30, 40, 50 percent; 100 runs per cell per
  mix; rung 1 at 20 runs and excluded.
- Faults: timeout, garbled_payload, error_response, empty_result, equal weights, nested
  across rates, applied to every tool including submit_record.
- Spend cap: none needed; snapshot 1 has no hosted model.

## 3. Decision rules (mirroring r3, sharpened in r4)

### Success

A run succeeds when its grader score is at least tau. Tau is 0.8 and the dataset is
difficulty level 4, chosen by calibration on 2026-09-26 so that the anchor cell (2 tools, 0
faults) lands in the band [0.15, 0.95] for every local model: gpt-oss-20b 26 of 30,
Qwen3-14B 12 of 30, Qwen3-4B 6 of 30 in the pilot. (Confirmed 2026-09-26.) The band was
[0.70, 0.90] in r3. The anchor study (`docs/method-note.md` Section 6) found the verdicts
hold for anchors from 0.12 to 0.97 and fail at 0.998, where a real edge is detected 1.2
percent of the time, so r4 widens the band to [0.15, 0.95], inside the tested range. Levels
1 to 3 put all three models at 100 percent at the anchor; level 4 prints no amounts and
states how they are computed. A model with a malformed tool-call rate above 20 percent at
the anchor is not a replicate.

### Intervals

Per cell: Wilson score interval at 95 percent. Between two cells: Newcombe hybrid score
interval (1998, method 10). The chart draws non-overlap of Wilson intervals, which is stricter.

### P1

One logistic regression per model, mix A, in-scope runs (tools 2 to 6):
logit P(success) = b0 + b1 tc + b2 tc2 + b3 f, with tc = tools - 4, tc2 = tc^2, f the fault
rate, fitted at run level with cluster-robust covariance by document. Fitted peak
t* = 4 - b1 / (2 b2), defined only when b2 < 0.

- C1 curvature: b2 < 0 with its cluster-robust Wald interval excluding 0.
- C2 peak: the cluster bootstrap percentile interval for t* (2,000 document resamples) lies
  inside [2, 6].
- C3 drop: in the zero-fault row, the Newcombe interval for p(peak column) - p(6 tools)
  excludes 0, where the peak column is round(t*) clipped to [2, 6]. Without a peak
  (b2 >= 0) the column is 2, so the drop is measured across the whole range; C3 then fails
  by definition but the drop can still restrict. (Confirmed 2026-09-26.)
- Each bootstrap resample is refitted through the same ladder as the primary fit (MLE,
  then Firth when the MLE fails the separation rule). Resamples without a peak (b2 >= 0),
  and resamples whose Firth refit also fails, are counted, not dropped: each enters the t*
  percentiles as minus infinity when b1 <= 0 and plus infinity when b1 > 0. The interval's
  limits are the 51st of 2,000 draws from each end, so more than 2.5 percent of such draws
  on one side opens that end. The peak column is t* rounded half up. (Confirmed
  2026-09-26.)
- Categories are assigned in the order not estimable, blocked, survives, restricted, below
  resolution; the first whose rule holds wins. Confirmed 2026-09-26, including
  that survives wins when the whole drop interval lies below the SESOI.

Survives when C1, C2 and C3 hold. Restricted when b2's interval lies entirely above 0, or
t*'s interval lies entirely outside [2, 6], or the drop interval excludes 15 points
(SESOI) on the small side. Below resolution when b2's interval includes 0 and the drop
interval includes both 0 and the SESOI; any case matching none of the named branches is also
below resolution, recorded with the condition `residual_branch` (confirmed 2026-09-26;
its frequency under simulated flat truths is in the method note). Blocked when the
fault-cap check fails (b3 above 0 with an interval excluding 0). In the Firth rung, C1 and
the fault-cap check read bootstrap percentile intervals instead of Wald intervals. Fit ladder: MLE, then Firth on non-convergence or any
|b_j| > 15, then `not_estimable` (a distinct category) when the in-scope outcome has no
variation.

### P2

Population: faulted runs in in-scope cells of mix A that recovered. Control: the run with
the same document, epoch and tool column at fault 0. Statistic R = mean of turns(faulted) -
turns(control). Interval: cluster bootstrap over documents, 2,000 resamples, percentile.
Survives when the lower bound is above 0; restricted when the interval includes 0 and the
upper bound is below 1.0 turn (SESOI), an interval entirely below 0 included;
below resolution otherwise; not estimable when no recovered faulted run has a control
(confirmed 2026-09-26). The secondary
statistic (extra turns minus fault count) is reported and moves nothing in snapshot 1.

### Instrument check

If the anchor cell's observed rate falls outside the calibration band by more than its
Wilson half-width, the snapshot is `instrument_out_of_calibration`: descriptives publish,
neither proposition moves.

### Descriptive only

Viable fraction, boundary width, fault-axis shape (LIN versus LOG3 by AIC), recovery in
seconds and by fault kind, sensitivity at tau plus and minus 0.1, fractional logit on the
continuous score, faults per run per cell, malformed-call rate.

## 4. Noise floor

Local: the cell (3 tools, 20 percent) rerun 50 times with fixed seeds on a 20-document block
in the sweep's own regime, one llama.cpp slot with requests queued in round-robin order
(identity rate, flip rate with Wilson interval, jitter SD), and 50 times with varied seeds
over the full cell (SD of p_r beside the binomial SD, ratio phi). Hosted: three
cells (2, 0), (4, 20), (6, 40), each rerun 50 times on a fixed 10-document block (inert in snapshot 1, which has no hosted model).
Jitter is reported beside the intervals, never inside them.

## 5. Departures from r3 recorded in r4

1. Hosted noise floor uses the block reading (10 documents per rerun), not the full cell.
2. tau and difficulty are set by calibration in a discarded pilot, not fixed by fiat.
3. Sampling above zero (temperature 0.7 locally, provider default hosted) with derived seeds.
4. Faults apply to submit_record as well.
5. Rung 1 is run but excluded from every statistic by construction.
6. The fit ladder ends in a named `not_estimable` category distinct from below resolution.
7. The local stack is llama.cpp with one server slot and prompt caching off, not a
   batch-invariant vLLM server. Determinism is claimed only if the three-regime gate passes;
   otherwise the stack is labelled non-deterministic and the floor is published as jitter.
   Because a single slot cannot vary batch composition, the fixed-seed floor is a check and
   shrinks to 50 reruns of a 20-document block. (Confirmed 2026-09-26.)
8. Each local model samples at its vendor's published recommendation, and every llama.cpp
   sampler is set explicitly so no server default applies. (Confirmed 2026-09-26.)
9. The calibration band is [0.15, 0.95], not [0.70, 0.90], on the anchor study's evidence.
   (Confirmed 2026-09-26.)
10. The task gains difficulty level 4 (computed amounts), and snapshot 1 uses it: levels 1
    to 3 put every model at 100 percent at the anchor. (Confirmed 2026-09-26.)
11. Qwen3-4B joins as a third local replicate, the same family and quantization as
    Qwen3-14B. (Confirmed 2026-09-26.)

## 6. Deposit checklist

- [x] tau 0.8, difficulty 4 and band [0.15, 0.95] written into `r4.yaml` and `snapshot-1.yaml`
- [x] replicate list confirmed: gpt-oss-20b, Qwen3-14B, Qwen3-4B (all passed the gate)
- [x] mix B decision stated: both mixes run; verdicts on mix A
- [x] spend cap per hosted snapshot: not needed, snapshot 1 is local only
- [x] SESOI values confirmed (15 points, 1.0 turn)
- [x] M3 rule completions confirmed (`docs/method-note.md` Section 5): the resample
  ladder, no-peak and failed resamples, the no-peak drop column, half-up rounding, the order
  of categories, the named residual branch, P2 below zero, and P2 not estimable
- [x] hosted block size: not needed, snapshot 1 is local only
- [x] `islands freeze` run on a clean tree; lock committed; tag v1.0.0 (local)
- [x] `islands probe` and `estimate`: not needed, local runs cost nothing
- [ ] repository public (after the deposit; the lock ties the code to the deposit)
- [x] Zenodo deposit of `deposit/bundle.zip` (`islands deposit`), published 2026-09-26; DOI recorded above
- [ ] deposit date is earlier than the first snapshot run (noise floor included)
