"""The six offline, deterministic tools of the public invoice-extraction task.

Every tool here is a pure function of (document, arguments): no network, no clock, no
filesystem beyond the dataset the run context already holds, no state that survives a run.
Two of them are required by the task (``fetch_document`` and ``submit_record``); the other
four are the distractors the sweep adds in mix order (ARCHITECTURE.md Section 7):

    rung 3  search_web        a plausible alternative to fetching the document
    rung 4  calculator        irrelevant to the task
    rung 5  read_spreadsheet  a near-duplicate of fetch_document with a different rendering
    rung 6  send_email        plausible but wrong

The harness never calls these functions directly. The only path from a model's tool call to
one of these coroutines is ``islands_harness.tools.execute``, wrapped by
``islands_harness.faults.FaultInjector``. This module is hashed by ``islands freeze``: the
JSON schemas below are part of what the model sees, so editing a description after the
freeze changes the snapshot.

Signature contract for every tool: ``async def execute(args: dict, ctx: RunContext) -> str``.
The return value is the exact string placed in the tool result message.
"""

from __future__ import annotations

import csv
import decimal
import io
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from islands_harness import rng
from islands_harness.dataset import SEARCH_INDEX_NAME, SPREADSHEET_COLUMNS, SPREADSHEETS_NAME
from islands_harness.tools import RunContext, ToolSpec

# --------------------------------------------------------------------------------------
# Required tools
# --------------------------------------------------------------------------------------


async def fetch_document(args: dict[str, Any], ctx: RunContext) -> str:
    """Return the full text of the document named by ``doc_id``.

    Pure function of (document, args). A ``doc_id`` that is not the run's document returns a
    frozen not-found JSON error rather than another document, so a run can never read
    across documents.
    """
    if args.get("doc_id") != ctx.doc_id:
        return json.dumps({"error": "not_found", "doc_id": args.get("doc_id")}, sort_keys=True)
    return ctx.document_text


async def submit_record(args: dict[str, Any], ctx: RunContext) -> str:
    """Store the submitted record on the run context and acknowledge it.

    The first accepted submission ends the run (loop.py). Fault semantics live in faults.py:
    the three unexecuted fault kinds never reach this function, so nothing is stored and the
    model must resubmit; the garbled kind reaches it, stores the record, and garbles only the
    acknowledgement string returned here.
    """
    record = args.get("record")
    ctx.record_submission(record)
    return json.dumps(
        {"status": "accepted", "fields": sorted(record) if isinstance(record, dict) else []},
        sort_keys=True,
    )


# --------------------------------------------------------------------------------------
# Distractor tools
# --------------------------------------------------------------------------------------


SEARCH_RESULTS_SHOWN = 3


async def search_web(args: dict[str, Any], ctx: RunContext) -> str:
    """Return canned search results for ``query`` from a fixed offline index.

    The index is generated with the dataset (dataset.py) and holds, per document, a handful
    of plausible but unhelpful hits: the vendor's fictional homepage snippet, a generic
    invoice-format article, and a hit that mentions a different invoice number. Results are a
    deterministic function of (document, query): the hits shown are chosen by
    ``rng.unit("search", doc_id, query)`` over the document's canned list. Nothing here is
    fetched from anywhere.

    Selection: with L hits in the document's list, u = rng.unit("search", doc_id, query) picks
    the start s = floor(u * L), and the result is the SEARCH_RESULTS_SHOWN hits at positions
    s, s + 1, ... taken cyclically (all L when L is smaller). The query is used exactly as
    given. Output: ``{"query": ..., "results": [{"snippet", "title", "url"}, ...]}`` as JSON
    with sorted keys; a document absent from the index gets an empty result list.
    """
    query = args.get("query")
    if not isinstance(query, str):
        query = str(query)
    index_path = Path(ctx.dataset_dir) / SEARCH_INDEX_NAME
    index: dict[str, list[dict[str, str]]] = json.loads(index_path.read_text(encoding="utf-8"))
    hits = index.get(ctx.doc_id, [])
    shown: list[dict[str, str]] = []
    if hits:
        start = int(rng.unit("search", ctx.doc_id, query) * len(hits))
        count = min(SEARCH_RESULTS_SHOWN, len(hits))
        shown = [hits[(start + k) % len(hits)] for k in range(count)]
    return json.dumps({"query": query, "results": shown}, sort_keys=True)


# Calculator limits and frozen replies. The limits keep recursion bounded; the context is
# explicit so no global decimal setting can change a result.
CALCULATOR_MAX_LENGTH = 1000
CALCULATOR_MAX_DEPTH = 50
CALCULATOR_CONTEXT = decimal.Context(
    prec=28,
    rounding=decimal.ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    traps=[decimal.DivisionByZero, decimal.Overflow, decimal.InvalidOperation],
)
CALCULATOR_PARSE_ERROR = json.dumps(
    {
        "error": "invalid_expression",
        "message": "Only numbers, + - * / and parentheses are allowed.",
    },
    sort_keys=True,
)
CALCULATOR_ARITHMETIC_ERROR = json.dumps(
    {"error": "arithmetic_error", "message": "The expression cannot be evaluated."},
    sort_keys=True,
)

_CALC_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]*)?|\.[0-9]+")
_CALC_OPERATORS = "+-*/()"


class _CalcSyntaxError(Exception):
    """The expression is not in the calculator grammar."""


def _calc_tokens(expression: str) -> list[str]:
    """Split into number and operator tokens; anything else is a syntax error."""
    tokens: list[str] = []
    pos = 0
    while pos < len(expression):
        char = expression[pos]
        if char.isspace():
            pos += 1
        elif char in _CALC_OPERATORS:
            tokens.append(char)
            pos += 1
        elif match := _CALC_NUMBER.match(expression, pos):
            tokens.append(match.group())
            pos = match.end()
        else:
            raise _CalcSyntaxError(char)
    return tokens


class _CalcParser:
    """Recursive descent over the grammar

        expression := term (("+" | "-") term)*
        term       := factor (("*" | "/") factor)*
        factor     := ("+" | "-") factor | number | "(" expression ")"

    Arithmetic happens in whatever decimal context is active; calculator() sets it.
    """

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.depth = 0

    def parse(self) -> Decimal:
        value = self.expression()
        if self.pos != len(self.tokens):
            raise _CalcSyntaxError("trailing input")
        return value

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise _CalcSyntaxError("unexpected end")
        self.pos += 1
        return token

    def expression(self) -> Decimal:
        value = self.term()
        while self.peek() in ("+", "-"):
            operator = self.take()
            right = self.term()
            value = value + right if operator == "+" else value - right
        return value

    def term(self) -> Decimal:
        value = self.factor()
        while self.peek() in ("*", "/"):
            operator = self.take()
            right = self.factor()
            value = value * right if operator == "*" else value / right
        return value

    def factor(self) -> Decimal:
        self.depth += 1
        if self.depth > CALCULATOR_MAX_DEPTH:
            raise _CalcSyntaxError("nested too deeply")
        token = self.take()
        if token == "-":
            value = -self.factor()
        elif token == "+":
            value = +self.factor()
        elif token == "(":
            value = self.expression()
            if self.take() != ")":
                raise _CalcSyntaxError("unclosed parenthesis")
        elif token in _CALC_OPERATORS:
            raise _CalcSyntaxError(token)
        else:
            value = Decimal(token)
        self.depth -= 1
        return value


async def calculator(args: dict[str, Any], ctx: RunContext) -> str:
    """Evaluate an arithmetic ``expression`` and return the result as a decimal string.

    Grammar: numbers, + - * / and parentheses; nothing else. Evaluated with ``decimal`` at 28
    digits, never with ``eval``. A parse error returns a frozen JSON error. Irrelevant to the
    task on purpose: the invoice total is printed in the document.

    Numbers are ASCII digits with an optional decimal point (no exponents, signs are the
    unary operators). Expressions longer than CALCULATOR_MAX_LENGTH characters or nested
    deeper than CALCULATOR_MAX_DEPTH are parse errors. Division by zero and overflow return
    CALCULATOR_ARITHMETIC_ERROR. The result is written in plain notation (``format(value,
    "f")``), keeping the Decimal's exponent, so 2.50 * 2 is ``5.00``; negative zero is ``0``.
    """
    expression = args.get("expression")
    if not isinstance(expression, str) or len(expression) > CALCULATOR_MAX_LENGTH:
        return CALCULATOR_PARSE_ERROR
    try:
        tokens = _calc_tokens(expression)
        with decimal.localcontext(CALCULATOR_CONTEXT):
            value = _CalcParser(tokens).parse()
            if value.is_zero():
                value = abs(value)
    except _CalcSyntaxError:
        return CALCULATOR_PARSE_ERROR
    except decimal.DecimalException:
        return CALCULATOR_ARITHMETIC_ERROR
    return format(value, "f")


def _not_found(doc_id: Any) -> str:
    """The frozen not-found reply, identical to fetch_document's."""
    return json.dumps({"error": "not_found", "doc_id": doc_id}, sort_keys=True)


async def read_spreadsheet(args: dict[str, Any], ctx: RunContext) -> str:
    """Return the document's line items rendered as CSV, a near-duplicate of fetch_document.

    Rendering: a header row ``description,quantity,unit_price,amount`` followed by one row per
    line item as generated (before any layout noise the difficulty knob adds to the text
    rendering), values quoted with the csv module's minimal quoting. The header fields of the
    invoice (number, date, vendor, currency, total) are NOT in the CSV, so a model that uses
    this tool instead of fetch_document cannot fill the whole record. Same not-found rule as
    fetch_document.

    The rows come from the dataset's spreadsheets.json (the generated line items without
    labels or header fields; RunContext carries no such structure), with LF line endings.
    A document absent from that file gets the not-found reply.
    """
    doc_id = args.get("doc_id")
    if doc_id != ctx.doc_id:
        return _not_found(doc_id)
    tables_path = Path(ctx.dataset_dir) / SPREADSHEETS_NAME
    tables: dict[str, list[dict[str, str]]] = json.loads(tables_path.read_text(encoding="utf-8"))
    rows = tables.get(ctx.doc_id)
    if rows is None:
        return _not_found(doc_id)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SPREADSHEET_COLUMNS)
    for row in rows:
        writer.writerow([row[column] for column in SPREADSHEET_COLUMNS])
    return buffer.getvalue()


async def send_email(args: dict[str, Any], ctx: RunContext) -> str:
    """Pretend to queue an email and return a deterministic acknowledgement.

    Nothing is sent anywhere. The message id is a hash of (doc_id, to, subject), so the same
    call in a replay returns the same acknowledgement. Plausible but wrong for the task: no
    record is stored, and the run still ends without a submission.
    """
    from islands_harness.rng import u32

    message_id = (
        f"msg-{u32('email', ctx.doc_id, str(args.get('to')), str(args.get('subject'))):08x}"
    )
    return json.dumps({"status": "queued", "message_id": message_id}, sort_keys=True)


# --------------------------------------------------------------------------------------
# Specs and registry
# --------------------------------------------------------------------------------------

_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        "vendor_name": {"type": "string"},
        "currency": {"type": "string", "description": "ISO 4217 code"},
        "total_amount": {"type": "number"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "amount": {"type": "number"},
                },
                "required": ["description", "quantity", "unit_price", "amount"],
            },
        },
    },
    "required": [
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "currency",
        "total_amount",
        "line_items",
    ],
}

FETCH_DOCUMENT = ToolSpec(
    name="fetch_document",
    description="Fetch the full text of a document by its id.",
    parameters={
        "type": "object",
        "properties": {"doc_id": {"type": "string", "description": "The document id."}},
        "required": ["doc_id"],
    },
    execute=fetch_document,
)

SUBMIT_RECORD = ToolSpec(
    name="submit_record",
    description="Submit the extracted invoice record. Call exactly once with the complete record.",
    parameters={
        "type": "object",
        "properties": {"record": _RECORD_SCHEMA},
        "required": ["record"],
    },
    execute=submit_record,
)

SEARCH_WEB = ToolSpec(
    name="search_web",
    description="Search the web and return the top results for a query.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    execute=search_web,
)

CALCULATOR = ToolSpec(
    name="calculator",
    description="Evaluate an arithmetic expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "For example (12.5 * 3) + 4"}
        },
        "required": ["expression"],
    },
    execute=calculator,
)

READ_SPREADSHEET = ToolSpec(
    name="read_spreadsheet",
    description="Read a document's tabular data as CSV by its id.",
    parameters={
        "type": "object",
        "properties": {"doc_id": {"type": "string", "description": "The document id."}},
        "required": ["doc_id"],
    },
    execute=read_spreadsheet,
)

SEND_EMAIL = ToolSpec(
    name="send_email",
    description="Send an email message.",
    parameters={
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
    execute=send_email,
)

# The registry the harness loads (islands_harness.tools.Registry.load). Order here is not
# the mix order; mixes come from the config. Names must be unique.
REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        FETCH_DOCUMENT,
        SUBMIT_RECORD,
        SEARCH_WEB,
        CALCULATOR,
        READ_SPREADSHEET,
        SEND_EMAIL,
    )
}
