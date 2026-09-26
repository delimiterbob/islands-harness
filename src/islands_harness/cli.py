"""The ``islands`` command line: argument parsing and dispatch, nothing else.

Every subcommand maps to one function in one module (ARCHITECTURE.md Section 6). This file
holds no harness logic; a handler loads the config, calls the module function and prints its
result. Keeping it that way means a reviewer can find the code behind any command from the
dispatch table below.

Exit codes:
    0  success
    1  an error the harness reported (exception message on stderr)
    2  usage error (argparse)
    3  lock mismatch: a frozen input changed (config.LockMismatch)
    4  spend cap reached before the phase finished
    5  doctor found a blocking problem
    6  not implemented yet (a TODO milestone marker was reached)

Run commands accept ``--exploratory``: with it, the lock is not checked and every output is
written under ``results/exploratory/`` and labelled so; nothing exploratory can enter a
snapshot. Without it, after ``freeze`` every run command refuses a changed hash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_LOCK, EXIT_CAP, EXIT_DOCTOR, EXIT_TODO = 0, 1, 2, 3, 4, 5, 6

DEFAULT_CONFIG = "configs/snapshot-1.yaml"


def _add_common(p: argparse.ArgumentParser, *, run_command: bool = False) -> None:
    p.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG, help="snapshot config YAML (default: %(default)s)"
    )
    if run_command:
        p.add_argument(
            "--exploratory",
            action="store_true",
            help="skip the lock check; outputs go to results/exploratory/",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="islands", description="Islands of Stability harness")
    sub = parser.add_subparsers(dest="command", required=True)

    dr = sub.add_parser(
        "doctor", help="check toolchain, VRAM, keys, endpoint, parser, outbound hosts"
    )
    dr.add_argument("--model", help="check only this model id")
    dr.add_argument("--rehash", action="store_true", help="re-hash weights, ignoring the cache")
    _add_common(dr)

    ds = sub.add_parser("dataset", help="generate or verify the task dataset").add_subparsers(
        dest="dataset_cmd", required=True
    )
    g = ds.add_parser("generate")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--difficulty", type=int, default=2)
    _add_common(g)
    _add_common(ds.add_parser("verify"))

    mo = sub.add_parser("models", help="fetch and verify local model weights").add_subparsers(
        dest="models_cmd", required=True
    )
    f = mo.add_parser("fetch", help="download each local model's weights at the pinned revision")
    f.add_argument("--verify", action="store_true", help="then re-hash every file")
    _add_common(f)
    _add_common(mo.add_parser("verify", help="re-hash every local weights file against its lock"))

    sv = sub.add_parser(
        "serve", help="start or stop the pinned llama-server for one local model"
    ).add_subparsers(dest="serve_cmd", required=True)
    s = sv.add_parser("start")
    s.add_argument("model", help="gpt-oss-20b or qwen3-14b; runs server/llamacpp/serve.ps1")
    s.add_argument(
        "--flash-attn",
        choices=["on", "off"],
        default="on",
        help="off only for the gate's flash_attn_off variant",
    )
    s.add_argument("--skip-hash-check", action="store_true", help="skip serve.ps1's weights hash")
    _add_common(s)
    st_ = sv.add_parser("stop")
    st_.add_argument("model", nargs="?", help="stop only this model's server")
    _add_common(st_)
    _add_common(sv.add_parser("status"))

    for name, run_cmd in (
        ("smoke", True),
        ("calibrate", True),
        ("verify-determinism", True),
        ("probe", True),
        ("noise-floor", True),
    ):
        p = sub.add_parser(name)
        p.add_argument("--model", required=name != "calibrate", help="model id from the config")
        if name == "calibrate":
            p.add_argument("--levels", type=int, nargs="*", help="difficulty levels to pilot")
            p.add_argument(
                "--recommend",
                action="store_true",
                help="choose one difficulty and tau from every model's calibration.json",
            )
        if name == "smoke":
            p.add_argument(
                "--doc-index",
                type=int,
                default=0,
                help="which run index (and so which document and seed) to use; default %(default)s",
            )
        if name == "verify-determinism":
            p.add_argument(
                "--variant",
                default="default",
                help="label for a gate rerun with changed server flags, for example flash_attn_off",
            )
        _add_common(p, run_command=run_cmd)

    dp = sub.add_parser(
        "deposit", help="assemble the dated pre-registration deposit (local; nothing is uploaded)"
    )
    dp.add_argument("--out", type=Path, default=Path("deposit"))
    _add_common(dp)

    fz = sub.add_parser("freeze", help="hash every fixed input and write the lock")
    fz.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be locked and what is still unconfirmed; write nothing",
    )
    _add_common(fz)

    e = sub.add_parser("estimate", help="spend estimate from a probe forecast")
    e.add_argument("--model", required=True)
    e.add_argument("--from", dest="forecast", required=True, help="forecast.json from `probe`")
    _add_common(e)

    sw = sub.add_parser("sweep", help="run the grid; appends to runs.jsonl; never prints a verdict")
    sw.add_argument("--model", required=True)
    sw.add_argument("--mix", default="A")
    sw.add_argument("--resume", action="store_true")
    sw.add_argument(
        "--cells", nargs="*", help="restrict to cells as TOOLS:FAULT, for example 3:0.2"
    )
    _add_common(sw, run_command=True)

    a = sub.add_parser("analyze", help="runs.jsonl + frozen prereg -> results.json")
    a.add_argument("results", type=Path)
    a.add_argument(
        "--check", action="store_true", help="recompute and diff an existing results.json"
    )
    a.add_argument(
        "--exploratory",
        action="store_true",
        help="skip the lock check and label the snapshot <id>-exploratory",
    )
    _add_common(a)

    r = sub.add_parser(
        "report", help="results.json -> snapshot.json, heatmap.svg, statement.txt, manifest.json"
    )
    r.add_argument("results", type=Path, help="a model directory or a snapshot directory")
    r.add_argument("--kind", choices=["snapshot", "exploratory", "selftest"], default="snapshot")
    r.add_argument("--label", help="the snapshot label shown on the site")
    r.add_argument("--png", action="store_true", help="also write heatmap.png (plots extra)")
    _add_common(r)

    v = sub.add_parser(
        "verify", help="reviewer command: hashes, re-grade, re-analyze, optionally re-execute"
    )
    v.add_argument("results", type=Path)
    v.add_argument("--regrade", action="store_true")
    v.add_argument("--reexecute", action="store_true")
    v.add_argument("--cells", nargs="*", help="cells to re-execute, as TOOLS:FAULT")
    v.add_argument("--out", type=Path, help="where re-executed runs go (default: beside RESULTS)")
    v.add_argument(
        "--exploratory", action="store_true", help="skip the freeze lock check (unfrozen results)"
    )
    _add_common(v)

    rp = sub.add_parser(
        "replay", help="re-execute one run from its recorded spec and diff the transcript"
    )
    rp.add_argument("run_id")
    rp.add_argument("--results", type=Path, required=True)
    _add_common(rp)

    st = sub.add_parser("selftest", help="the whole pipeline on the fake provider")
    st.add_argument("--out", type=Path, default=Path("results/selftest"))
    _add_common(st)

    b = sub.add_parser("bundle", help="Zenodo bundle with SHA256SUMS")
    b.add_argument("results", type=Path)
    _add_common(b)

    sc = sub.add_parser("schema", help="export the JSON Schemas (config.v1, runs.v1, snapshot.v1)")
    sc.add_argument("--out", type=Path, default=Path("schemas"))

    cl = sub.add_parser("client", help="client mode").add_subparsers(
        dest="client_cmd", required=True
    )
    ci = cl.add_parser("init")
    ci.add_argument("dir", type=Path)
    cr = cl.add_parser("run")
    cr.add_argument("dir", type=Path)
    cr.add_argument("--redact", action="store_true")
    cr.add_argument(
        "--quick",
        action="store_true",
        help="4x4 at 30 runs; labelled below resolution by construction",
    )

    d = sub.add_parser("distance", help="distance from an operating point to the nearest edge")
    d.add_argument("--tools", type=int, required=True)
    d.add_argument("--fault", type=float, required=True)
    d.add_argument("--results", type=Path, required=True)
    d.add_argument("--target", type=float, default=0.9, help="the client's target success rate")
    _add_common(d)

    return parser


# -- handlers: load, call one module function, print ------------------------------------


def _config(args: argparse.Namespace):  # noqa: ANN202
    from islands_harness.config import load_config

    return load_config(args.config)


def _cmd_doctor(args: argparse.Namespace) -> int:
    from islands_harness.provenance import doctor

    report = doctor(_config(args), _repo_root(args), model_id=args.model, rehash=args.rehash)
    print(report.render())
    return EXIT_DOCTOR if report.blocking else EXIT_OK


def _cmd_models(args: argparse.Namespace) -> int:
    from islands_harness import localstack

    cfg, root = _config(args), _repo_root(args)
    failures = 0
    for model in (m for m in cfg.models if m.expect is not None):
        if args.models_cmd == "fetch":
            print(f"{model.id}: {localstack.fetch_weights(root, model)}")
        if args.models_cmd == "verify" or getattr(args, "verify", False):
            ok, detail = localstack.verify_weights(root, model, rehash=True)
            print(f"{model.id}: {'ok' if ok else 'FAILED'}: {detail}")
            failures += not ok
    return EXIT_OK if not failures else EXIT_ERROR


def _cmd_serve(args: argparse.Namespace) -> int:
    from islands_harness import localstack

    cfg, root = _config(args), _repo_root(args)
    if args.serve_cmd == "start":
        model, log = localstack.serve_start(
            cfg, args.model, root, flash_attn=args.flash_attn, skip_hash_check=args.skip_hash_check
        )
        print(f"{model.id} is serving on {model.base_url} (log {log})")
        _, diffs, _ = localstack.server_differences(model, flash_attn=args.flash_attn)
        for line in diffs:
            print(f"  differs from the config: {line}")
        return EXIT_DOCTOR if diffs else EXIT_OK
    if args.serve_cmd == "stop":
        pids = localstack.serve_stop(cfg, args.model)
        print(f"stopped {len(pids)} llama-server process(es)")
        return EXIT_OK
    processes = localstack.server_processes()
    if not processes:
        print("no llama-server is running")
    for proc in processes:
        name = next(
            (
                m.id
                for m in cfg.models
                if m.expect is not None and localstack.model_port(m) == proc.port
            ),
            "unknown model",
        )
        print(f"pid {proc.pid}: {name} on port {proc.port}, -fa {proc.flags.get('-fa')}")
    return EXIT_OK


def _check_local_server(model) -> int | None:  # noqa: ANN001
    """EXIT code when a local model's server is down or differs from the config; None when
    it matches (or the model is hosted)."""
    from islands_harness import localstack

    if model.expect is None:
        return None
    running, diffs, _ = localstack.server_differences(model)
    if not running:
        print(
            f"{model.id} is not running; start it with `islands serve start {model.model}`",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if diffs:
        print("the running server differs from the config:", file=sys.stderr)
        for line in diffs:
            print(f"  {line}", file=sys.stderr)
        return EXIT_DOCTOR
    return None


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """The pilot at the anchor cell per difficulty level (one model), or with --recommend the
    joint choice of one difficulty and one tau from every model's calibration.json."""
    import json

    from islands_harness.provenance import calibrate, recommend_calibration
    from islands_harness.stats.verdict import jsonable

    cfg, root = _config(args), _repo_root(args)
    if args.recommend:
        results = []
        for m in cfg.models:
            path = _results_dir(args, cfg, root, m.id, "calibration") / "calibration.json"
            if path.is_file():
                results.append(json.loads(path.read_text(encoding="utf-8")))
        if not results:
            print(
                "no calibration.json yet; run `islands calibrate --model ID` first", file=sys.stderr
            )
            return EXIT_USAGE
        rec = recommend_calibration(results, cfg)
        out = (
            _results_dir(args, cfg, root, results[0]["model"], "calibration").parent.parent
            / "calibration-recommendation.json"
        )
        out.write_text(
            json.dumps(jsonable(rec), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"models piloted: {', '.join(r['model'] for r in results)}")
        for o in rec["options"]:
            rates = ", ".join(f"{m} {100 * e['p']:.0f}%" for m, e in o["models"].items())
            print(
                f"  difficulty {o['difficulty']}, tau {o['tau']:g}: {len(o['replicates'])} in band ({rates})"
            )
        chosen = rec["chosen"]
        if chosen is None:
            print("no option covers every piloted model")
            return EXIT_ERROR
        print(
            f"recommended: difficulty {chosen['difficulty']}, tau {chosen['tau']:g}; replicates "
            f"{chosen['replicates'] or 'none'}; excluded {chosen['excluded'] or 'none'} -> {out}"
        )
        return EXIT_OK
    if not args.model:
        print("calibrate needs --model, or --recommend", file=sys.stderr)
        return EXIT_USAGE
    model = cfg.model_by_id(args.model)
    code = _check_local_server(model)
    if code is not None:
        return code
    out = _results_dir(args, cfg, root, model.id, "calibration")
    result = calibrate(
        cfg,
        model,
        repo_root=root,
        out_dir=out,
        levels=args.levels,
        progress=lambda line: print(f"  {line}", flush=True),
    )
    band = cfg.calibration.target_band
    print(
        f"{model.id}: anchor cell, {cfg.calibration.pilot_runs} pilot runs per level, band {band[0]:g} to {band[1]:g}"
    )
    for level, lvl in sorted(result["levels"].items()):
        cells = "; ".join(
            f"tau {t}: {c['k']}/{c['n']} [{100 * c['lo']:.0f}, {100 * c['hi']:.0f}]{' in band' if c['in_band'] else ''}"
            for t, c in lvl["taus"].items()
        )
        print(
            f"  difficulty {level}: {cells}; malformed calls {100 * lvl['malformed_call_rate']:.1f}%"
        )
    print(f"calibration: {out / 'calibration.json'}")
    return EXIT_OK


def _cmd_verify_determinism(args: argparse.Namespace) -> int:
    """The three-regime gate against the running server, which must match the config (or
    the named variant) exactly before a single run starts."""
    from islands_harness import localstack
    from islands_harness.provenance import verify_determinism

    cfg, root = _config(args), _repo_root(args)
    model = cfg.model_by_id(args.model)
    if model.expect is None:
        print(
            "the gate runs on local models; a hosted model's jitter is the noise floor's job",
            file=sys.stderr,
        )
        return EXIT_USAGE
    flash = "off" if args.variant == "flash_attn_off" else None
    running, diffs, evidence = localstack.server_differences(model, flash_attn=flash)
    if not running:
        hint = " --flash-attn off" if flash else ""
        print(
            f"{model.id} is not running; start it with `islands serve start {model.model}{hint}`",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if diffs:
        print("the running server differs from the config:", file=sys.stderr)
        for line in diffs:
            print(f"  {line}", file=sys.stderr)
        return EXIT_DOCTOR
    out = _results_dir(args, cfg, root, model.id, "sweep")
    gate = cfg.determinism_gate
    print(
        f"gate: {model.id}, variant {args.variant}: {gate.specs} specs x {gate.replays_per_regime} "
        f"replays x 3 regimes, {gate.filler_requests} filler requests -> {out / 'determinism.json'}"
    )
    result = verify_determinism(
        cfg,
        model,
        repo_root=root,
        out_dir=out,
        variant=args.variant,
        server=evidence,
        gpu=localstack.gpu_memory(),
        progress=lambda line: print(f"  {line}", flush=True),
    )
    print(
        f"{result.label}: {result.executions} executions in {result.seconds:.0f} s, "
        f"{len(result.disagreeing_specs)} of {result.specs} specs disagreeing, "
        f"outcomes {result.outcomes}, filler failures {result.filler_failures}"
    )
    for d in result.divergences:
        print(
            f"  {d['run_id']}: {d['distinct_transcripts']} distinct transcripts, first differing message {d['first_divergent_message']}"
        )
    return EXIT_OK if result.passed else EXIT_ERROR


def _cmd_dataset(args: argparse.Namespace) -> int:
    from islands_harness import dataset

    cfg = _config(args)
    out = cfg.task.dir / cfg.task.dataset
    if args.dataset_cmd == "generate":
        dataset.generate(args.seed, args.n, args.difficulty, out)
    else:
        dataset.verify_lock(out)
        print(f"dataset lock verified: {out}")
    return EXIT_OK


def _repo_root(args: argparse.Namespace) -> Path:
    """The harness root: the config lives in <root>/configs/."""
    return Path(args.config).resolve().parent.parent


def _cmd_selftest(args: argparse.Namespace) -> int:
    from islands_harness.provenance import selftest

    summary = selftest(_config(args), args.out, _repo_root(args))
    for name, outcome in summary["outcomes"].items():
        print(f"  {name:<22} {outcome}")
    snap = summary["snapshot"]
    print(
        f"  synthetic snapshot     {snap['runs']} runs; re-analysis identical; "
        f"{snap['regraded']} re-graded; {snap['replays']} replays identical; "
        f"{snap['bundle_members']} files bundled and hash-checked"
    )
    print(f"  {snap['statement']}")
    print(f"selftest passed: {summary['runs']} scenario runs in {summary['out_dir']}")
    print(f"snapshot.json: {snap['snapshot_json']}")
    return EXIT_OK


def _cmd_smoke(args: argparse.Namespace) -> int:
    from islands_harness.provenance import smoke

    cfg = _config(args)
    model = cfg.model_by_id(args.model)
    out = _repo_root(args) / "results" / "exploratory" / "smoke" / model.id
    rows = smoke(cfg, model, out, _repo_root(args), doc_index=args.doc_index)
    print(
        f"smoke: {model.id} on {rows[0]['doc_id'] if rows else '?'} (exploratory; never part of a snapshot)"
    )
    for r in rows:
        score = "n/a" if r["score"] is None else f"{r['score']:.2f}"
        print(
            f"  {r['cell']:<20} {r['outcome']:<17} turns {r['turns']:>2}  calls {r['tool_calls']:>2}  "
            f"faults {r['faults']:>2}  malformed {r['malformed']:>2}  score {score:>4}  {r['seconds']:>6.1f} s  {r['tokens']:>6} tokens"
        )
    print(f"transcripts: {out}")
    return EXIT_OK


def _require_frozen(args: argparse.Namespace, cfg, root: Path) -> None:  # noqa: ANN001
    """Run commands refuse to produce snapshot data from an unfrozen or changed setup.
    ``--exploratory`` skips the check and routes every output to results/exploratory/."""
    if getattr(args, "exploratory", False):
        return
    from islands_harness.config import check_lock, read_lock

    lock_path = root / "configs" / f"{Path(args.config).stem}.lock.json"
    if not lock_path.exists():
        raise RuntimeError(
            f"no freeze lock at {lock_path}; run `islands freeze` first, or pass --exploratory"
        )
    check_lock(cfg, read_lock(lock_path), root)


def _results_dir(args: argparse.Namespace, cfg, root: Path, model_id: str, phase: str) -> Path:  # noqa: ANN001
    if getattr(args, "exploratory", False):
        return root / "results" / "exploratory" / phase / model_id
    base = root / cfg.snapshot.outputs / model_id
    return base if phase == "sweep" else base / phase


def _every(n: int):  # noqa: ANN202
    """A progress printer that speaks every n runs: counts and spend only, never a verdict."""
    seen = {"k": 0}

    def show(line: str) -> None:
        seen["k"] += 1
        if seen["k"] % n == 0:
            print(f"  {line}", flush=True)

    return show


def _cmd_sweep(args: argparse.Namespace) -> int:
    import asyncio

    from islands_harness.config import load_prereg
    from islands_harness.providers.factory import build_provider
    from islands_harness.runner import SpendLedger, Storage, load_prices, load_task, run_specs
    from islands_harness.specs import expand_cells
    from islands_harness.specs import run_specs as specs_for_cell

    cfg, root = _config(args), _repo_root(args)
    model = cfg.model_by_id(args.model)
    _require_frozen(args, cfg, root)
    task = load_task(cfg, root)
    wanted = None
    if args.cells:  # TOOLS:FAULT pairs, for example 3:0.2
        wanted = {(int(t), round(float(f), 6)) for t, f in (c.split(":") for c in args.cells)}
    cells = [
        c
        for c in expand_cells(cfg, args.mix)
        if wanted is None or (c.tools, round(c.fault_rate, 6)) in wanted
    ]
    specs = [s for c in cells for s in specs_for_cell(cfg, model, c, sorted(task.documents))]
    out = _results_dir(args, cfg, root, model.id, "sweep")
    ledger = (
        SpendLedger(
            prices=load_prices(root / model.prices, model.model), cap_usd=model.spend_cap_usd
        )
        if model.prices
        else None
    )
    tau = load_prereg(root / cfg.snapshot.preregistration).success.tau
    print(
        f"sweep: {model.id}, mix {args.mix}, {len(cells)} cells, {len(specs)} runs planned -> {out}"
    )

    async def go():  # noqa: ANN202
        provider = build_provider(model, cfg.agent)
        try:
            return await run_specs(
                specs,
                provider,
                task,
                Storage(out),
                ledger,
                config=cfg,
                model=model,
                tau=tau,
                resume=args.resume,
                progress=_every(25),
            )
        finally:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()

    summary = asyncio.run(go())
    print(
        f"sweep done: {summary.executed} executed, {summary.skipped_existing} already present, "
        f"{summary.reexecuted} re-executed after a transport abort, {summary.aborted_transport} aborted twice"
        + (f", spend ${summary.spend_usd:.2f}" if ledger else "")
    )
    if summary.stopped_at_cap:
        print(
            "stopped at 95 percent of the spend cap; nothing further was started", file=sys.stderr
        )
        return EXIT_CAP
    return EXIT_OK


def _cmd_probe(args: argparse.Namespace) -> int:
    import datetime

    from islands_harness.provenance import probe
    from islands_harness.providers.factory import build_provider

    cfg, root = _config(args), _repo_root(args)
    model = cfg.model_by_id(args.model)
    _require_frozen(args, cfg, root)
    out = _results_dir(args, cfg, root, model.id, "probe")
    forecast = probe(
        cfg,
        model,
        build_provider(model, cfg.agent),
        repo_root=root,
        out_dir=out,
        measured_on=datetime.date.today().isoformat(),
        progress=_every(25),
    )
    print(
        f"probe: {model.id}, {sum(p['runs'] for p in forecast.points)} usable runs over {len(forecast.points)} points; "
        f"mean per run {forecast.mean_input_tokens:.0f} in, {forecast.mean_output_tokens:.0f} out, "
        f"{forecast.mean_cache_read_tokens:.0f} cache read, {forecast.mean_turns:.1f} turns"
    )
    print(f"forecast: {out / 'forecast.json'}")
    return EXIT_OK


def _cmd_estimate(args: argparse.Namespace) -> int:
    import json

    from islands_harness.provenance import Forecast, estimate, phase_counts
    from islands_harness.runner import load_prices

    cfg, root = _config(args), _repo_root(args)
    model = cfg.model_by_id(args.model)
    if model.prices is None:
        print(f"{model.id} has no price table (a local model); its spend is zero", file=sys.stderr)
        return EXIT_OK
    forecast = Forecast(**json.loads(Path(args.forecast).read_text(encoding="utf-8")))
    plan = estimate(
        forecast,
        load_prices(root / model.prices, model.model),
        phase_counts(cfg, model),
        model.spend_cap_usd,
        price_table=str(model.prices),
    )
    for phase in plan.runs:
        print(f"  {phase:<18} {plan.runs[phase]:>6} runs  ${plan.usd[phase]:>9.2f}")
    cap = (
        f"cap ${plan.cap_usd:.2f}, stop line ${0.95 * plan.cap_usd:.2f}"
        if plan.cap_usd
        else "no cap"
    )
    print(
        f"  total             ${plan.total_usd:.2f}  ({cap}): {'within' if plan.within_cap else 'OVER'}"
    )
    return EXIT_OK if plan.within_cap else EXIT_CAP


def _gate_pairing(results: Path) -> str:
    """How exact the P2 pairing is on a local stack: exact only when the determinism gate
    recorded in the results directory verified the stack."""
    import json

    gate = results / "determinism.json"
    if gate.is_file():
        label = json.loads(gate.read_text(encoding="utf-8")).get("label")
        if label == "deterministic_verified":
            return "exact_invariant_stack"
    return "in_expectation_unverified_stack"


def _cmd_analyze(args: argparse.Namespace) -> int:
    from islands_harness.config import load_prereg, prereg_hash
    from islands_harness.provenance import is_hosted
    from islands_harness.stats.analyze import analyze, read_runs

    if args.check:
        return _analysis_check(args)
    cfg, root = _config(args), _repo_root(args)
    _require_frozen(args, cfg, root)
    results = Path(args.results)
    runs_path = results / "runs.jsonl"
    if not runs_path.is_file():
        print(f"no runs.jsonl in {results}", file=sys.stderr)
        return EXIT_USAGE
    try:
        runs, torn = read_runs(runs_path)
    except ValueError as exc:
        print(f"islands analyze: {exc}", file=sys.stderr)
        return EXIT_ERROR
    ids = sorted({str(r.get("model_id")) for r in runs})
    if len(ids) != 1:
        print(f"runs.jsonl must hold one model's runs; found {ids}", file=sys.stderr)
        return EXIT_USAGE
    model = cfg.model_by_id(ids[0])
    prereg = load_prereg(root / cfg.snapshot.preregistration)
    snapshot_id = cfg.snapshot.id + ("-exploratory" if args.exploratory else "")
    payload = analyze(
        runs,
        prereg,
        snapshot_id=snapshot_id,
        model_id=model.id,
        pairing="in_expectation_hosted" if is_hosted(model) else _gate_pairing(results),
        prereg_hash=prereg_hash(prereg),
        out_dir=results,
        torn_lines=torn,
    )
    print(payload["statement"])
    print(f"results: {results / 'results.json'}")
    return EXIT_OK


def _print_check(check) -> None:  # noqa: ANN001
    for f in check.files:
        print(f"  {f.name:<26} {f.status}")
        for line in f.differences[:5]:
            print(f"      {line}")
    if not check.prereg_matches:
        print(f"  {check.detail}")


def _analysis_check(args: argparse.Namespace) -> int:
    """analyze --check: recompute into a temporary directory and diff with the published
    files; exit 0 when every file is identical or equal within tolerance."""
    from islands_harness.verification import check_analysis

    cfg, root = _config(args), _repo_root(args)
    check = check_analysis(Path(args.results), config=cfg, repo_root=root)
    _print_check(check)
    print("analyze --check: passed" if check.ok else "analyze --check: FAILED")
    return EXIT_OK if check.ok else EXIT_ERROR


def _lock_dict(args: argparse.Namespace, root: Path) -> dict | None:
    import json

    path = root / "configs" / f"{Path(args.config).stem}.lock.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _cmd_report(args: argparse.Namespace) -> int:
    from islands_harness.report import model_dirs, report, write_heatmap_png

    cfg, root = _config(args), _repo_root(args)
    results = Path(args.results)
    models = None
    if args.kind == "selftest":
        from islands_harness.provenance import selftest_model

        synthetic = selftest_model(cfg)
        models = {synthetic.id: synthetic}
    snapshot = report(
        results,
        config=cfg,
        repo_root=root,
        kind=args.kind,
        label=args.label,
        models=models,
        lock=_lock_dict(args, root) if args.kind == "snapshot" else None,
    )
    if args.png:
        for d, entry in zip(model_dirs(results), snapshot.models, strict=True):
            print(f"png: {write_heatmap_png(entry.grid.model_dump(), d / 'heatmap.png')}")
    for entry in snapshot.models:
        print(entry.statement)
    print(f"snapshot: {results / 'snapshot.json'}")
    return EXIT_OK


def _cmd_bundle(args: argparse.Namespace) -> int:
    from islands_harness.report import bundle

    result = bundle(Path(args.results))
    print(f"bundle: {result.path} ({result.members} members)")
    print(f"sums: {result.sums_path}")
    print(f"bundle sha256 {result.sha256} (informational; SHA256SUMS is the reference)")
    return EXIT_OK


async def _with_provider(model, agent, work):  # noqa: ANN001, ANN201
    """Build the live provider, run ``work(provider)`` and close the provider, all in one
    event loop."""
    from islands_harness.providers.factory import build_provider

    provider = build_provider(model, agent)
    try:
        return await work(provider)
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()


def _cmd_replay(args: argparse.Namespace) -> int:
    import asyncio

    from islands_harness.provenance import replay_async
    from islands_harness.stats.analyze import read_runs

    cfg, root = _config(args), _repo_root(args)
    results = Path(args.results)
    rows = {r["run_id"]: r for r in read_runs(results / "runs.jsonl")[0]}
    if args.run_id not in rows:
        print(f"no run {args.run_id} in {results / 'runs.jsonl'}", file=sys.stderr)
        return EXIT_USAGE
    model = cfg.model_by_id(str(rows[args.run_id]["model_id"]))
    diff = asyncio.run(
        _with_provider(
            model,
            cfg.agent,
            lambda provider: replay_async(
                args.run_id, results, provider, config=cfg, repo_root=root, model=model
            ),
        )
    )
    print(f"{diff.run_id}: {diff.detail}")
    print(f"  outcome     {diff.original_outcome} -> {diff.replay_outcome}")
    print(f"  transcript  {diff.original_hash[:16]} -> {diff.replay_hash[:16]}")
    return EXIT_OK if diff.identical else EXIT_ERROR


def _cmd_verify(args: argparse.Namespace) -> int:
    """The reviewer command: bundle hashes, the freeze lock, re-analysis of every model
    directory, optionally re-grading and re-execution. Exit 0 only when every check passes."""
    import asyncio
    import json

    from islands_harness.config import check_lock, read_lock
    from islands_harness.report import SUMS_NAME, check_sums, model_dirs
    from islands_harness.verification import check_analysis, reexecute, regrade

    cfg, root = _config(args), _repo_root(args)
    results = Path(args.results)
    failures = 0
    if (results / SUMS_NAME).is_file():
        sums = check_sums(results)
        bad = sums["mismatched"] + sums["missing"]
        print(f"SHA256SUMS: {'verified' if not bad else 'FAILED'}")
        for name in bad[:10]:
            print(f"  {name}")
        if sums["unlisted"]:
            print(f"  {len(sums['unlisted'])} file(s) added after the bundle (not checked)")
        failures += bool(bad)
    else:
        print("SHA256SUMS: none in this directory (not a bundle); file hashes not checked")
    lock_path = root / "configs" / f"{Path(args.config).stem}.lock.json"
    if args.exploratory:
        print("lock: not checked (--exploratory)")
    elif lock_path.is_file():
        check_lock(cfg, read_lock(lock_path), root)  # raises LockMismatch: exit code 3
        print(f"lock: every fixed input matches {lock_path.name}")
    else:
        print(f"lock: {lock_path.name} not found; pass --exploratory for unfrozen results")
        failures += 1
    dirs = model_dirs(results)
    if not dirs:
        print(f"no results.json under {results}", file=sys.stderr)
        return EXIT_USAGE
    for d in dirs:
        check = check_analysis(d, config=cfg, repo_root=root)
        print(f"re-analysis of {d.name}: {'passed' if check.ok else 'FAILED'}")
        _print_check(check)
        failures += not check.ok
        if args.regrade:
            graded = regrade(d, config=cfg, repo_root=root)
            print(
                f"re-grading of {d.name}: {graded.checked} runs, "
                f"{'passed' if graded.ok else 'FAILED'}"
            )
            for line in (
                graded.score_mismatches + graded.submission_mismatches + graded.missing_transcripts
            )[:10]:
                print(f"  {line}")
            failures += not graded.ok
    if args.reexecute:
        if not args.cells:
            print("--reexecute needs --cells TOOLS:FAULT ...", file=sys.stderr)
            return EXIT_USAGE
        cells = [(int(t), float(f)) for t, f in (c.split(":") for c in args.cells)]
        for d in dirs:
            model = cfg.model_by_id(
                json.loads((d / "results.json").read_text(encoding="utf-8"))["model"]
            )
            out = args.out or results.parent / f"{results.name}-reexecute"
            report = asyncio.run(
                _with_provider(
                    model,
                    cfg.agent,
                    lambda provider, d=d, model=model, out=out: reexecute(
                        d,
                        cells,
                        provider,
                        config=cfg,
                        model=model,
                        repo_root=root,
                        out_dir=Path(out) / d.name,
                    ),
                )
            )
            for r in report:
                lo, hi = r.difference_interval
                print(
                    f"re-execution {d.name} {r.tools}:{r.fault:g}: published {r.published_k}/"
                    f"{r.published_n}, now {r.k}/{r.n}, difference [{100 * lo:.0f}, {100 * hi:.0f}] "
                    f"points, identical transcripts {r.identical_transcripts}/{r.compared_transcripts}"
                    f"{'' if r.consistent else '  INCONSISTENT'}"
                )
                failures += not r.consistent
    print("verify: passed" if not failures else f"verify: {failures} check(s) failed")
    return EXIT_OK if not failures else EXIT_ERROR


CONFIRM_MARKER = "CONFIRM BEFORE FREEZE"


def unconfirmed_items(args: argparse.Namespace, root: Path) -> list[str]:
    """Every line still marked CONFIRM BEFORE FREEZE in the pre-registration, its prose
    mirror and the snapshot config, as "file:line: text"."""
    cfg = _config(args)
    files = [
        root / cfg.snapshot.preregistration,
        root / "PREREGISTRATION.md",
        Path(args.config).resolve(),
    ]
    import re

    # The marker may wrap across lines in prose (and carry Markdown bold), so match its three
    # words separated by any whitespace, and report the line where it starts.
    pattern = re.compile(r"CONFIRM\s+BEFORE\s+FREEZE")
    found = []
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for match in pattern.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            found.append(
                f"{path.relative_to(root).as_posix()}:{number}: {lines[number - 1].strip()[:110]}"
            )
    return found


def _cmd_freeze(args: argparse.Namespace) -> int:
    import datetime
    import subprocess

    from islands_harness.config import freeze, write_lock

    root = _repo_root(args)
    pending = unconfirmed_items(args, root)
    if args.dry_run:
        cfg = _config(args)
        lock = freeze(cfg, root, frozen_on=datetime.date.today().isoformat(), git_commit="dry-run")
        print(
            f"would lock: config {lock.config[:12]}, pre-registration {lock.preregistration[:12]}, "
            f"system prompt {lock.system_prompt[:12]}, {len(lock.tool_schemas)} tool lists, "
            f"{len(lock.dataset)} dataset files, grader, loop, faults, providers, tools"
        )
        print(f"{len(pending)} item(s) still marked {CONFIRM_MARKER}:")
        for item in pending:
            print(f"  {item}")
        return EXIT_OK
    if pending:
        print(
            f"freeze refused: {len(pending)} item(s) still marked {CONFIRM_MARKER}.",
            file=sys.stderr,
        )
        print(
            "Confirm or change each one, remove its marker, and commit; then freeze.",
            file=sys.stderr,
        )
        for item in pending[:20]:
            print(f"  {item}", file=sys.stderr)
        return EXIT_USAGE
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"freeze needs a git repository with at least one commit at {root}"
        ) from exc
    if status.strip():
        raise RuntimeError("freeze refuses a dirty tree; commit or stash every change first")
    cfg = _config(args)
    lock = freeze(cfg, root, frozen_on=datetime.date.today().isoformat(), git_commit=commit)
    if not lock.dataset:
        raise RuntimeError("freeze refuses an empty dataset; run `islands dataset generate` first")
    path = root / "configs" / f"{Path(args.config).stem}.lock.json"
    write_lock(lock, path)
    print(
        f"froze {len(lock.dataset)} dataset files, {len(lock.tool_schemas)} tool lists at commit {commit[:12]}: {path}"
    )
    return EXIT_OK


def _cmd_deposit(args: argparse.Namespace) -> int:
    from islands_harness.report import REDACTIONS_NAME, deposit_files, prepare_deposit

    cfg, root = _config(args), _repo_root(args)
    config_path = Path(args.config).resolve()
    result = prepare_deposit(cfg, root, config_path, args.out)
    for f in deposit_files(cfg, root, config_path):
        print(f"  {f.resolve().relative_to(root.resolve()).as_posix()}")
    if (Path(args.out) / REDACTIONS_NAME).is_file():
        print(f"  {REDACTIONS_NAME} (local paths shortened in the copies it lists)")
    print(f"deposit: {result.path} ({result.members} members); upload it to Zenodo yourself")
    return EXIT_OK


def _cmd_schema(args: argparse.Namespace) -> int:
    from islands_harness.report import export_schemas

    for name, path in export_schemas(args.out).items():
        print(f"{name}: {path}")
    return EXIT_OK


def _not_wired(milestone: str):  # noqa: ANN202
    def handler(args: argparse.Namespace) -> int:
        print(f"islands {args.command}: not implemented yet ({milestone})", file=sys.stderr)
        return EXIT_TODO

    return handler


# One entry per subcommand. Handlers added as milestones land; ARCHITECTURE.md Section 16.
DISPATCH = {
    "doctor": _cmd_doctor,
    "dataset": _cmd_dataset,
    "models": _cmd_models,
    "serve": _cmd_serve,
    "smoke": _cmd_smoke,
    "calibrate": _cmd_calibrate,
    "freeze": _cmd_freeze,
    "verify-determinism": _cmd_verify_determinism,
    "probe": _cmd_probe,
    "estimate": _cmd_estimate,
    "noise-floor": _not_wired("TODO(M7): provenance.noise_floor"),
    "sweep": _cmd_sweep,
    "analyze": _cmd_analyze,
    "report": _cmd_report,
    "verify": _cmd_verify,
    "replay": _cmd_replay,
    "selftest": _cmd_selftest,
    "bundle": _cmd_bundle,
    "schema": _cmd_schema,
    "deposit": _cmd_deposit,
    "client": _not_wired("TODO(M9): client mode"),
    "distance": _not_wired("TODO(M9): distance"),
}


def main(argv: list[str] | None = None) -> int:
    # Windows code pages never touch a transcript or a printed hash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return DISPATCH[args.command](args)
    except NotImplementedError as exc:
        print(f"islands {args.command}: {exc}", file=sys.stderr)
        return EXIT_TODO
    except Exception as exc:  # noqa: BLE001
        from islands_harness.config import LockMismatch

        if isinstance(exc, LockMismatch):
            print(f"lock mismatch: {exc}", file=sys.stderr)
            return EXIT_LOCK
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
