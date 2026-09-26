"""Cells, run specs and the paired design.

A sweep is a grid of cells (tool count, fault rate, mix). Each cell expands into run specs,
and run index ``i`` in every cell maps to the same document and the same sampling seed, so
two cells differ only in the tool list offered and the fault rate applied. That pairing is
what makes cell contrasts and the P2 control matching work (ARCHITECTURE.md Sections 3
and 7): the control for a faulted run is the run with the same index in the same tool
column at fault 0.

Everything here is a pure function of the config and the root seed. Documents are
permuted once per model with a hash-derived Fisher-Yates so the order is stable and
reviewable; index ``i`` maps to ``permutation[i % D]`` at epoch ``i // D``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from islands_harness.config import ModelConfig, SnapshotConfig
from islands_harness.rng import seed64, unit


@dataclass(frozen=True, order=True)
class Cell:
    tools: int
    fault_rate: float
    mix: str

    @property
    def in_scope(self) -> bool:
        return self.tools >= 2

    def label(self) -> str:
        return f"{self.mix}:{self.tools}:{self.fault_rate:.1f}"


@dataclass(frozen=True)
class RunSpec:
    """Everything the loop needs to execute one run, and everything replay needs to repeat it."""

    run_id: str
    model_id: str
    cell: Cell
    index: int
    doc_id: str
    epoch: int
    sample_seed: int
    tool_list: tuple[str, ...]
    phase: str = "sweep"  # sweep | rung1 | probe | gate | floor_fixed | floor_varied | floor_hosted | calibrate | smoke
    seed_offset: int | None = None  # varied-seed floor reruns; None otherwise


def run_id(model_id: str, cell: Cell, index: int, phase: str, seed_offset: int | None) -> str:
    """Stable id: the first 16 hex chars of sha256 over the spec's identity fields.

    The id never includes the document id or seed directly because those are functions of
    (model_id, index) already; two specs with the same id are the same run and resume skips
    the second.
    """
    ident = {
        "model_id": model_id,
        "mix": cell.mix,
        "tools": cell.tools,
        "fault_rate": cell.fault_rate,
        "index": index,
        "phase": phase,
        "seed_offset": seed_offset,
    }
    payload = json.dumps(ident, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def expand_cells(config: SnapshotConfig, mix: str) -> list[Cell]:
    """All cells of the sweep for one mix, rung 1 included (it is excluded downstream)."""
    if mix not in config.task.mixes:
        raise KeyError(f"mix {mix!r} not in config.task.mixes")
    return [
        Cell(tools=t, fault_rate=f, mix=mix)
        for t in config.sweep.tools_axis
        for f in config.sweep.fault_axis
    ]


def tool_list_for(config: SnapshotConfig, mix: str, k: int) -> tuple[str, ...]:
    """The tools offered at rung ``k``: rung 1 is the single rung1 tool; otherwise the first
    ``k`` of the mix, which starts with the two required tools."""
    if k == 1:
        return tuple(config.task.rung1)
    mix_tools = config.task.mixes[mix]
    if mix_tools[:2] != config.task.required_tools:
        raise ValueError(
            f"mix {mix!r} must start with the required tools {config.task.required_tools}"
        )
    if k > len(mix_tools):
        raise ValueError(f"rung {k} exceeds mix {mix!r} of {len(mix_tools)} tools")
    return tuple(mix_tools[:k])


def document_permutation(
    config: SnapshotConfig, model: ModelConfig, doc_ids: list[str]
) -> list[str]:
    """Hash-derived Fisher-Yates over the sorted document ids, one permutation per model.

    Draw ``j = floor(unit("perm", root_seed, model_id, i) * (i + 1))`` for ``i`` from
    ``n - 1`` down to 1 and swap positions ``i`` and ``j``. Pure and platform-stable.
    """
    order = sorted(doc_ids)
    root = config.snapshot.root_seed
    for i in range(len(order) - 1, 0, -1):
        j = int(unit("perm", root, model.id, i) * (i + 1))
        order[i], order[j] = order[j], order[i]
    return order


def run_specs(
    config: SnapshotConfig,
    model: ModelConfig,
    cell: Cell,
    doc_ids: list[str],
    *,
    phase: str = "sweep",
    n_runs: int | None = None,
    seed_offset: int | None = None,
) -> list[RunSpec]:
    """Expand one cell into run specs with the paired mapping of index to document and seed.

    ``n_runs`` defaults to ``documents * epochs`` for in-scope cells and
    ``sweep.rung1_runs_per_cell`` for rung 1. The sampling seed for index ``i`` is
    ``seed64("sample", root_seed, model_id, i)`` and does not depend on the cell; with a
    ``seed_offset`` (varied-seed floor) the scope gains ``("varied", seed_offset)``.
    """
    perm = document_permutation(config, model, doc_ids)[: model.documents]
    if not perm:
        raise ValueError("no documents")
    if n_runs is None:
        n_runs = config.sweep.rung1_runs_per_cell if cell.tools == 1 else len(perm) * model.epochs
    tools = tool_list_for(config, cell.mix, cell.tools)
    root = config.snapshot.root_seed
    specs: list[RunSpec] = []
    for i in range(n_runs):
        doc = perm[i % len(perm)]
        epoch = i // len(perm)
        seed = (
            seed64("sample", root, model.id, i)
            if seed_offset is None
            else seed64("sample", root, model.id, i, "varied", seed_offset)
        )
        specs.append(
            RunSpec(
                run_id=run_id(model.id, cell, i, phase, seed_offset),
                model_id=model.id,
                cell=cell,
                index=i,
                doc_id=doc,
                epoch=epoch,
                sample_seed=seed,
                tool_list=tools,
                phase=phase,
                seed_offset=seed_offset,
            )
        )
    return specs


def probe_specs(
    config: SnapshotConfig, model: ModelConfig, doc_ids: list[str], mix: str
) -> list[RunSpec]:
    """``probe.runs_per_point`` specs at each (tools, fault) probe point, phase ``probe``: the
    first ``runs_per_point`` indices of each point, so every point sees the same documents and
    seeds (the paired design), and run ids never collide with sweep ids."""
    specs: list[RunSpec] = []
    for tools in config.probe.tools_points:
        for fault in config.probe.fault_points:
            specs.extend(
                run_specs(
                    config,
                    model,
                    Cell(tools, fault, mix),
                    doc_ids,
                    phase="probe",
                    n_runs=config.probe.runs_per_point,
                )
            )
    return specs


def gate_specs(
    config: SnapshotConfig, model: ModelConfig, doc_ids: list[str], mix: str
) -> list[RunSpec]:
    """The determinism gate's fixed specs: the first ``determinism_gate.specs`` indices of
    the anchor cell (2 tools, 0 faults), phase ``gate``, each with its own document and
    derived seed. The three regimes replay these same specs (provenance.verify_determinism).
    """
    anchor = Cell(2, 0.0, mix)
    return run_specs(
        config, model, anchor, doc_ids, phase="gate", n_runs=config.determinism_gate.specs
    )


def noise_floor_specs(
    config: SnapshotConfig, model: ModelConfig, doc_ids: list[str], mix: str
) -> dict[str, list[list[RunSpec]]]:
    """Specs for the noise floor protocols, keyed by protocol, one inner list per rerun.

    ``fixed``: the local floor cell's full spec list repeated ``fixed_seed_reruns`` times
    (identical specs; the stack, not the harness, is what may vary).
    ``varied``: the same cell ``varied_seed_reruns`` times with ``seed_offset = rerun index``.
    ``hosted``: each hosted cell, ``reruns`` times, restricted to the first
    ``documents_per_rerun`` indices (the block reading recorded in r4).

    TODO(M7): implement on top of run_specs.
    """
    raise NotImplementedError("TODO(M7): noise floor specs")
