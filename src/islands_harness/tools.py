"""ToolSpec, the registry, mix ordering and the single execute path.

``execute()`` is the only function in the harness that turns a model's tool call into a
tool implementation's result (ARCHITECTURE.md Section 7). ``faults.FaultInjector`` wraps
it; ``loop.run_one`` calls the injector and nothing else. No other code may call a
``ToolSpec.execute`` coroutine, so a reviewer auditing the boundary reads this file, faults.py
and loop.py and is done.

Argument validation happens here, against the tool's JSON schema, and is deliberately
simple and frozen: a malformed call gets ``MALFORMED_ARGUMENTS_TEXT`` back, tagged
``model_error`` in the event log, and is never repaired or retried. Tools are sent to the
model without ``strict`` on every provider (D11), so what the model emits is the measurement.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from islands_harness.providers.base import ToolCall, ToolResult

ToolExecute = Callable[[dict[str, Any], "RunContext"], Awaitable[str]]

# Frozen catalog string returned for malformed arguments. Hashed with this file by freeze.
# Deliberately generic: it names no field, so the model cannot learn the schema from it.
MALFORMED_ARGUMENTS_TEXT = json.dumps(
    {"error": "invalid_arguments", "message": "The arguments did not match the tool's schema."},
    sort_keys=True,
)

UNKNOWN_TOOL_TEXT = json.dumps(
    {"error": "unknown_tool", "message": "No tool with that name is available."}, sort_keys=True
)


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what the model sees (name, description, parameters) and what runs (execute)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object
    execute: ToolExecute

    def schema(self) -> dict[str, Any]:
        """The provider-neutral schema dict; adapters wrap it in their wire format."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class RunContext:
    """Per-run state the tools and the injector can see. Never contains the gold record."""

    doc_id: str
    document_text: str
    dataset_dir: Path
    tools: dict[str, ToolSpec]  # the rung's offered tools, by name
    turn: int = 0
    call_index: int = 0  # 0-based across all tool calls in execution order
    submission: dict[str, Any] | None = None
    submission_accepted: bool = False
    submissions: list[Any] = field(default_factory=list)
    redact: bool = False

    def record_submission(self, record: Any) -> None:
        """Store a submitted record. Called only from submit_record's execute path."""
        self.submissions.append(record)
        if isinstance(record, dict) and not self.submission_accepted:
            self.submission = record
            self.submission_accepted = True


class Registry:
    """The task's tools plus the mix ordering from the config."""

    def __init__(
        self,
        specs: dict[str, ToolSpec],
        *,
        required: list[str],
        mixes: dict[str, list[str]],
        rung1: list[str],
    ) -> None:
        missing = [n for m in mixes.values() for n in m if n not in specs] + [
            n for n in rung1 if n not in specs
        ]
        if missing:
            raise KeyError(
                f"tools named in the config but absent from the registry: {sorted(set(missing))}"
            )
        self.specs = dict(specs)
        self.required = list(required)
        self.mixes = {k: list(v) for k, v in mixes.items()}
        self.rung1 = list(rung1)

    @classmethod
    def load(
        cls,
        target: str | Path,
        *,
        required: list[str],
        mixes: dict[str, list[str]],
        rung1: list[str],
    ) -> Registry:
        """Load ``REGISTRY`` from a dotted module path or a ``.py`` file path.

        The public task lives outside the package (tasks/invoice_extraction/tools.py) and is
        loaded by file path; a client's tools may be either.
        """
        if isinstance(target, Path) or str(target).endswith(".py"):
            module = load_module_from_path(Path(target))
        else:
            module = importlib.import_module(str(target))
        registry = getattr(module, "REGISTRY", None)
        if not isinstance(registry, dict):
            raise ImportError(f"{target} defines no REGISTRY dict")
        return cls(registry, required=required, mixes=mixes, rung1=rung1)

    def for_rung(self, mix: str, k: int) -> list[ToolSpec]:
        """The tools offered at rung ``k`` of ``mix``: the two required tools first, then the
        mix order; rung 1 is the single rung1 tool. The list, in this order, is what every
        provider sends, so its hash is part of the lock."""
        if k == 1:
            names = self.rung1
        else:
            order = self.mixes[mix]
            if order[:2] != self.required:
                raise ValueError(f"mix {mix!r} must start with {self.required}")
            names = order[:k]
        return [self.specs[n] for n in names]

    def schema_hash(self, tools: list[ToolSpec]) -> str:
        """sha256 over the canonical JSON of the schemas as ordered. Recorded per rung by freeze."""
        payload = json.dumps(
            [t.schema() for t in tools], separators=(",", ":"), sort_keys=True, ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_module_from_path(path: Path) -> Any:
    """Import a task file (tools, grader) by path with the standard importlib recipe.

    The module is registered in ``sys.modules`` before it executes, because dataclasses and
    other code that resolves annotations look the module up there; skipping that step breaks
    any task file that uses ``from __future__ import annotations``. The name carries a short
    hash of the resolved path so two tasks' files with the same stem never collide.
    """
    resolved = Path(path).resolve()
    name = f"islands_task_{resolved.stem}_{hashlib.sha256(str(resolved).encode('utf-8')).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def validate_arguments(schema: dict[str, Any], args: Any) -> bool:
    """Frozen, minimal JSON-schema check: object type, required keys, and the primitive
    ``type`` of each declared property (string, number, integer, boolean, array, object).
    Nested object properties are checked one level down; anything the schema does not
    declare is allowed. No third-party validator so the rule is small enough to read.

    Depth is counted from the arguments object (depth 0). Objects at depth 0 and 1 have their
    required keys and declared property types checked; an array at depth 0 or 1 has each
    element's type checked against ``items``. Nothing deeper is checked: for submit_record the
    record's fields are typed, the fields inside each line item are left to the grader.
    ``bool`` is never a number or an integer, and an integer-valued float is not an integer.
    """
    return _matches(schema, args, depth=0)


_PRIMITIVES: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _matches(schema: dict[str, Any], value: Any, depth: int) -> bool:
    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared] if declared else []
    if types and not any(_PRIMITIVES.get(t, lambda v: False)(value) for t in types):
        return False
    if isinstance(value, dict) and depth <= 1:
        if any(key not in value for key in schema.get("required", [])):
            return False
        for key, sub in schema.get("properties", {}).items():
            if key in value and not _matches(sub, value[key], depth + 1):
                return False
    if isinstance(value, list) and depth <= 1 and isinstance(schema.get("items"), dict):
        return all(_matches(schema["items"], item, depth + 1) for item in value)
    return True


def args_sha256(args: Any) -> str:
    payload = json.dumps(
        args, separators=(",", ":"), sort_keys=True, ensure_ascii=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def execute(call: ToolCall, ctx: RunContext) -> ToolResult:
    """The only path from a tool call to a tool implementation.

    Order of checks: unknown tool name -> UNKNOWN_TOOL_TEXT (model_error); malformed
    argument JSON or schema failure -> MALFORMED_ARGUMENTS_TEXT (model_error); otherwise the
    spec's coroutine runs with the parsed arguments and its string return is the result.
    Exceptions inside a tool are harness bugs and propagate (they never become model-visible
    text), because a silent catch here would be a hidden knob.
    """
    spec = ctx.tools.get(call.name)
    if spec is None:
        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=UNKNOWN_TOOL_TEXT,
            is_error=True,
            executed=False,
            model_error=True,
        )
    if (
        call.malformed
        or not isinstance(call.arguments, dict)
        or not validate_arguments(spec.parameters, call.arguments)
    ):
        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=MALFORMED_ARGUMENTS_TEXT,
            is_error=True,
            executed=False,
            model_error=True,
        )
    content = await spec.execute(call.arguments, ctx)
    return ToolResult(
        call_id=call.id, name=call.name, content=content, is_error=False, executed=True
    )
