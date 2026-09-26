"""Tests for the public task: the dataset generator, the grader and the six task tools.

The task files live outside the package and the harness loads them by path without
registering them in sys.modules (tools.Registry.load); the loader below does the same, so a
file that only works when imported as a package module fails here.

The strongest check is the careful-reader test: it transcribes every generated document
verbatim (no arithmetic, no reformatting) and requires the transcription to score exactly
1.0 against the gold. That proves the text and the gold agree, and that the frozen grader
accepts every date, amount and currency format the generator prints.
"""

from __future__ import annotations

import builtins
import csv
import importlib.util
import io
import json
import re
import shutil
import socket
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from islands_harness import dataset, rng
from islands_harness.config import LockMismatch
from islands_harness.tools import Registry, RunContext

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "tasks" / "invoice_extraction"
DATASET_DIR = TASK_DIR / "dataset"
COMMITTED = {"seed": 20261006, "n": 100, "difficulty": 4}
SMALL_SEED = 424242
SMALL_N = 60


def _load_task_module(name: str) -> ModuleType:
    """Load tasks/invoice_extraction/<name>.py by path, the way Registry.load does."""
    spec = importlib.util.spec_from_file_location(f"islands_task_{name}", TASK_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grader = _load_task_module("grader")
task_tools = _load_task_module("tools")


@pytest.fixture(scope="module")
def levels(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    """A small generated dataset at each difficulty level."""
    base = tmp_path_factory.mktemp("levels")
    out: dict[int, Path] = {}
    for level in dataset.DIFFICULTY_LEVELS:
        out[level] = base / f"level{level}"
        dataset.generate(SMALL_SEED, SMALL_N, level, out[level])
    return out


def _all_datasets(levels: dict[int, Path]) -> list[Path]:
    return [DATASET_DIR, *levels.values()]


def _ctx(doc_id: str, dataset_dir: Path = DATASET_DIR) -> RunContext:
    path = dataset_dir / "documents" / f"{doc_id}.txt"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return RunContext(
        doc_id=doc_id, document_text=text, dataset_dir=dataset_dir, tools=dict(task_tools.REGISTRY)
    )


# -- the careful reader ---------------------------------------------------------------------

_RECORD_FIELD = {
    "vendor": "vendor_name",
    "number": "invoice_number",
    "date": "invoice_date",
    "currency": "currency",
}
_HEADER_FIELD_BY_LABEL = {
    label: _RECORD_FIELD[block]
    for block, labels in dataset.LABELS.items()
    if block in _RECORD_FIELD
    for label in labels
}


def _read_document(text: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Transcribe a generated document verbatim: the record a careful reader would submit
    (every value a string exactly as printed) and the totals block as label -> printed value."""
    lines = text.split("\n")
    rules = [k for k, line in enumerate(lines) if line and set(line) == {"-"}]
    assert len(rules) == 2, "a document has exactly two table rules"
    record: dict[str, Any] = {}
    for line in lines[: rules[0] - 1]:
        label, sep, value = line.partition(": ")
        if sep and label in _HEADER_FIELD_BY_LABEL:
            record[_HEADER_FIELD_BY_LABEL[label]] = value
    computed = "Amount" not in lines[rules[0] - 1]  # level 4 prints no amounts
    items: list[dict[str, str]] = []
    for line in lines[rules[0] + 1 : rules[1]]:
        if line.startswith("  "):
            items[-1]["description"] += " " + line.strip()
            continue
        cells = re.split(r" {2,}", line.strip())
        item = {"description": cells[0], "quantity": cells[1], "unit_price": cells[2]}
        if computed:
            discount = int(cells[3].rstrip("%")) if len(cells) > 3 else 0
            price = grader.normalize_amount(cells[2])
            net = price * int(cells[1]) * (100 - discount) / 100
            item["amount"] = str(net.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP"))
        else:
            item["amount"] = cells[-1]
        items.append(item)
    record["line_items"] = items
    totals: dict[str, str] = {}
    block: list[str] = []
    for line in lines[rules[1] + 1 :]:
        if not line.strip():
            break
        block.append(line.strip())
        label, _, value = line.strip().partition(": ")
        totals[label] = value.strip()
    if computed:
        rate = Decimal(re.search(r" at ([\d.]+)% is charged", " ".join(block)).group(1))
        shipping = grader.normalize_amount(
            next(v for k, v in totals.items() if k in dataset.LABELS["shipping"])
        )
        subtotal = sum((Decimal(i["amount"]) for i in items), Decimal(0))
        tax = (subtotal * rate / 100).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        record["total_amount"] = str(subtotal + shipping + tax)
        return record, totals
    total_labels = [label for label in totals if label in dataset.LABELS["total"]]
    assert len(total_labels) == 1
    record["total_amount"] = totals[total_labels[0]]
    return record, totals


# -- gold and the committed dataset ------------------------------------------------------


def test_committed_dataset_lock_verifies() -> None:
    lock = dataset.verify_lock(DATASET_DIR)
    assert {k: lock[k] for k in COMMITTED} == COMMITTED
    assert len(lock["files"]) == COMMITTED["n"] + 3
    assert [d.id for d in dataset.load(DATASET_DIR)] == [f"inv-{i:04d}" for i in range(1, 101)]


def test_every_committed_gold_record_scores_one() -> None:
    gold = dataset.load_gold(DATASET_DIR)
    assert len(gold) == COMMITTED["n"]
    for doc_id, record in gold.items():
        result = grader.score(record.fields, record.fields)
        assert result.total == 1.0, doc_id
        assert set(result.per_field.values()) == {1.0}, doc_id


def test_careful_reading_of_every_document_scores_one(levels: dict[int, Path]) -> None:
    for dataset_dir in _all_datasets(levels):
        gold = dataset.load_gold(dataset_dir)
        for document in dataset.load(dataset_dir):
            record, _ = _read_document(document.text)
            result = grader.score(record, gold[document.id].fields)
            assert result.total == 1.0, (dataset_dir.name, document.id, result.notes)


def test_printed_amounts_add_up(levels: dict[int, Path]) -> None:
    """Items sum to the subtotal and subtotal + shipping + tax is the printed total."""
    for dataset_dir in _all_datasets(levels):
        for document in dataset.load(dataset_dir):
            if " Amount" not in document.text:
                continue  # level 4 prints no amounts to add up; the careful reading computes them
            record, totals = _read_document(document.text)
            item_sum = sum(grader.normalize_amount(i["amount"]) for i in record["line_items"])
            total = grader.normalize_amount(record["total_amount"])
            subtotal = grader.normalize_amount(totals.get("Subtotal", record["total_amount"]))
            assert item_sum == subtotal, document.id
            extras = [
                grader.normalize_amount(value)
                for label, value in totals.items()
                if label != "Subtotal" and label not in dataset.LABELS["total"]
            ]
            assert subtotal + sum(extras, Decimal(0)) == total, document.id


def test_invoice_number_is_printed_once_and_distractors_differ(levels: dict[int, Path]) -> None:
    for dataset_dir in _all_datasets(levels):
        gold = dataset.load_gold(dataset_dir)
        for document in dataset.load(dataset_dir):
            number = gold[document.id].fields["invoice_number"]
            assert document.text.count(number) == 1, document.id


def test_difficulty_levels_render_their_knobs(levels: dict[int, Path]) -> None:
    for level, dataset_dir in levels.items():
        gold = dataset.load_gold(dataset_dir)
        docs = {d.id: d.text for d in dataset.load(dataset_dir)}
        counts = [len(g.fields["line_items"]) for g in gold.values()]
        lo, hi = dataset.ITEM_COUNTS[level]
        assert lo <= min(counts) and max(counts) <= hi
        currencies = {g.fields["currency"] for g in gold.values()}
        has_subtotal = ["Subtotal:" in text for text in docs.values()]
        if level == 1:
            assert currencies == {"USD"}
            assert not any(has_subtotal)
            for doc_id, text in docs.items():
                assert f"Invoice date: {gold[doc_id].fields['invoice_date']}" in text
                assert "Previous balance" not in text and "Note:" not in text
        else:
            assert currencies == {"USD", "EUR", "GBP"}
            assert all(has_subtotal) if level in (2, 3) else not any(has_subtotal)
        if level == 2:
            for text in docs.values():
                po = any(f"{label}: " in text for label in dataset.LABELS["po_number"])
                assert po + ("Previous balance:" in text) == 1
        if level == 3:
            assert all("Note: our earlier invoice" in text for text in docs.values())
            assert all(" Disc. " in text for text in docs.values())
            assert any(re.search(r"^  \S", text, re.M) for text in docs.values()), "a wrap"
            assert any(re.search(r"\d\.\d{3},\d\d", text) for text in docs.values()), "EUR 1.234,56"
            assert any(re.search(r"\d,\d{3}\.\d\d", text) for text in docs.values()), "1,234.56"
            assert any(re.search(r" \d+%  ", text) for text in docs.values()), "a discount"
        if level == 4:
            assert all("Note: our earlier invoice" in text for text in docs.values())
            assert all(" Disc." in text and " Amount" not in text for text in docs.values())
            assert all(
                "is charged on the sum of the line amounts" in text for text in docs.values()
            )
            for doc_id, text in docs.items():
                total = gold[doc_id].fields["total_amount"]
                assert f"{total:,.2f}" not in text and f"{total:.2f}" not in text, doc_id


def test_slash_dates_are_never_ambiguous(levels: dict[int, Path]) -> None:
    for dataset_dir in _all_datasets(levels):
        for document in dataset.load(dataset_dir):
            for day, _month in re.findall(r"\b(\d\d)/(\d\d)/\d{4}\b", document.text):
                assert int(day) > 12, document.id


# -- generator determinism and refusals ----------------------------------------------------


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Every file's bytes by relative path, CRLF read as LF (a Windows checkout may convert)."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes().replace(b"\r\n", b"\n")
        for p in root.rglob("*")
        if p.is_file()
    }


def test_generator_same_seed_same_bytes(tmp_path: Path) -> None:
    dataset.generate(7, 12, 3, tmp_path / "a")
    dataset.generate(7, 12, 3, tmp_path / "b")
    assert _tree_bytes(tmp_path / "a") == _tree_bytes(tmp_path / "b")
    dataset.verify_lock(tmp_path / "a")
    dataset.generate(8, 12, 3, tmp_path / "c")
    assert _tree_bytes(tmp_path / "a") != _tree_bytes(tmp_path / "c")


def test_regenerating_the_committed_dataset_reproduces_it(tmp_path: Path) -> None:
    out = tmp_path / "again"
    dataset.generate(COMMITTED["seed"], COMMITTED["n"], COMMITTED["difficulty"], out)
    assert _tree_bytes(out) == _tree_bytes(DATASET_DIR)


def test_generate_refuses_an_existing_lock_and_stale_files(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    dataset.generate(1, 3, 1, out)
    with pytest.raises(FileExistsError):
        dataset.generate(1, 3, 1, out)
    (out / "LOCK.json").unlink()
    (out / "documents" / "inv-0999.txt").write_text("stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        dataset.generate(1, 3, 1, out)
    assert not (out / "LOCK.json").exists()


def test_generate_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        dataset.generate(1, 3, 5, tmp_path / "x")
    with pytest.raises(ValueError):
        dataset.generate(1, 0, 2, tmp_path / "y")


def test_verify_lock_catches_an_edit(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    dataset.generate(1, 3, 2, out)
    doc = out / "documents" / "inv-0002.txt"
    doc.write_text(doc.read_text(encoding="utf-8") + "x", encoding="utf-8")
    with pytest.raises(LockMismatch, match="inv-0002"):
        dataset.verify_lock(out)


# -- grader: normalizers --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-14", "2026-03-14"),
        ("2026-3-4", "2026-03-04"),
        ("14/03/2026", "2026-03-14"),
        ("03/14/2026", "2026-03-14"),  # MM/DD only because 14 > 12
        ("03/04/2026", "2026-04-03"),  # ambiguous resolves day-first
        ("14.03.2026", "2026-03-14"),
        ("14 March 2026", "2026-03-14"),
        ("March 14, 2026", "2026-03-14"),
        ("  MARCH 14,   2026 ", "2026-03-14"),
        ("20260314", "2026-03-14"),
        ("14/03/26", None),  # two-digit year
        ("2026-02-30", None),  # not on the calendar
        ("13/13/2026", None),
        ("Mar 14, 2026", None),  # abbreviations are not in the ladder
        ("14th March 2026", None),
        ("2026-03-14T00:00:00", None),
        ("", None),
        (None, None),
        (True, None),
    ],
)
def test_normalize_date(text: Any, expected: str | None) -> None:
    assert grader.normalize_date(text) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1,234.56", "1234.56"),
        ("1.234,56", "1234.56"),
        ("1 234,56", "1234.56"),
        ("$1,234.56", "1234.56"),
        ("1.234,56 \u20ac", "1234.56"),
        ("\u00a3-12.50", "-12.50"),
        ("EUR 12,50", "12.50"),
        ("12.50 usd", "12.50"),
        ("(12.50)", "-12.50"),
        ("-$12.50", "-12.50"),
        ("1,234", "1234"),
        ("12,5", "12.5"),
        ("1.234", "1.234"),
        (12.3, "12.3"),
        (7, "7"),
        (Decimal("5.00"), "5.00"),
        (None, None),
        (True, None),
        ("abc", None),
        ("", None),
        ("NaN", None),
        (float("nan"), None),
        (float("inf"), None),
        ("12.5.1", None),
        ("1e5", None),
        ("(-12.50)", None),
    ],
)
def test_normalize_amount(value: Any, expected: str | None) -> None:
    result = grader.normalize_amount(value)
    if expected is None:
        assert result is None
    else:
        assert result == Decimal(expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("usd", "USD"),
        (" EUR ", "EUR"),
        ("\u20ac", "EUR"),
        ("\u00a3", "GBP"),
        ("$", "USD"),
        ("Euro", ""),
        ("US$", ""),
        (840, ""),
        (None, ""),
    ],
)
def test_normalize_currency(value: Any, expected: str) -> None:
    assert grader.normalize_currency(value) == expected


# -- grader: hand-built failure cases -------------------------------------------------------

GOLD: dict[str, Any] = {
    "invoice_number": "QUI-2026-4413",
    "invoice_date": "2026-03-14",
    "vendor_name": "Quillhaven Office Supplies Inc.",
    "currency": "EUR",
    "total_amount": 1481.4,
    "line_items": [
        {
            "description": "A4 copy paper, 80 gsm, ream of 500",
            "quantity": 10,
            "unit_price": 4.25,
            "amount": 42.5,
        },
        {
            "description": "Ballpoint pens, blue, box of 50",
            "quantity": 3,
            "unit_price": 12.0,
            "amount": 36.0,
        },
        {
            "description": "Heavy duty stapler, full strip",
            "quantity": 2,
            "unit_price": 24.99,
            "amount": 49.98,
        },
        {
            "description": "Whiteboard markers, set of 8",
            "quantity": 5,
            "unit_price": 7.1,
            "amount": 35.5,
        },
    ],
}


def _with(**changes: Any) -> dict[str, Any]:
    record = json.loads(json.dumps(GOLD))
    record.update(changes)
    return record


def _items(*edits: tuple[int, str, Any]) -> list[dict[str, Any]]:
    items = json.loads(json.dumps(GOLD["line_items"]))
    for index, key, value in edits:
        items[index][key] = value
    return items


# Expected totals: weights are number 2, date 1, vendor 1, currency 1, total 2, items 3 (of 10).
GRADER_CASES: list[tuple[str, dict[str, Any], float]] = [
    ("exact", _with(), 1.0),
    ("wrong invoice number", _with(invoice_number="QUI-2026-4412"), 0.8),
    ("number case and spacing", _with(invoice_number="  qui-2026-4413 "), 1.0),
    ("distractor number", _with(invoice_number="QUI-2026-4377"), 0.8),
    ("wrong date", _with(invoice_date="2026-03-15"), 0.9),
    ("day-first date", _with(invoice_date="14/03/2026"), 1.0),
    ("long date", _with(invoice_date="March 14, 2026"), 1.0),
    ("two-digit year", _with(invoice_date="14/03/26"), 0.9),
    ("impossible date", _with(invoice_date="2026-02-30"), 0.9),
    ("vendor without trailing period", _with(vendor_name="quillhaven office supplies inc"), 1.0),
    ("vendor without legal suffix", _with(vendor_name="Quillhaven Office Supplies"), 0.9),
    ("currency lower case", _with(currency="eur"), 1.0),
    ("currency symbol of the gold", _with(currency="\u20ac"), 1.0),
    ("wrong currency", _with(currency="USD"), 0.9),
    ("symbol of another currency", _with(currency="$"), 0.9),
    ("currency name", _with(currency="Euro"), 0.9),
    ("total as printed string", _with(total_amount="1.481,40 \u20ac"), 1.0),
    ("total within tolerance", _with(total_amount=1481.404), 1.0),
    ("total off by a cent", _with(total_amount=1481.41), 0.8),
    ("subtotal instead of total", _with(total_amount=163.98), 0.8),
    ("total not a number", _with(total_amount="see document"), 0.8),
    ("items reordered", _with(line_items=list(reversed(GOLD["line_items"]))), 1.0),
    ("items as printed strings", _with(line_items=_items((0, "unit_price", "4,25"))), 1.0),
    ("missing line item", _with(line_items=GOLD["line_items"][:3]), 0.925),
    (
        "extra line item",
        _with(
            line_items=[
                *GOLD["line_items"],
                {"description": "Shipping", "quantity": 1, "unit_price": 9.95, "amount": 9.95},
            ]
        ),
        0.94,
    ),
    ("one wrong quantity", _with(line_items=_items((1, "quantity", 4))), 0.98125),
    ("one missing amount", _with(line_items=_items((1, "amount", None))), 0.98125),
    (
        "description with an extra word",
        _with(line_items=_items((0, "description", "A4 copy paper 80 gsm ream of 500 sheets"))),
        1.0,
    ),
    (
        "wrapped description cut at the boundary (Jaccard 3/5)",
        _with(line_items=_items((2, "description", "Heavy duty stapler,"))),
        1.0,
    ),
    (
        "description below the threshold (Jaccard 2/5)",
        _with(line_items=_items((2, "description", "Heavy stapler"))),
        0.925,
    ),
    ("line items not a list", _with(line_items="see document"), 0.7),
    ("line items empty", _with(line_items=[]), 0.7),
    ("a non-object item", _with(line_items=[*GOLD["line_items"][:3], "stapler"]), 0.925),
    ("missing currency field", {k: v for k, v in GOLD.items() if k != "currency"}, 0.9),
    ("empty dict", {}, 0.0),
]


@pytest.mark.parametrize(
    ("name", "submission", "expected"), GRADER_CASES, ids=[c[0] for c in GRADER_CASES]
)
def test_grader_cases(name: str, submission: dict[str, Any], expected: float) -> None:
    result = grader.score(submission, GOLD)
    assert result.total == pytest.approx(expected, abs=1e-12), result.notes
    assert 0.0 <= result.total <= 1.0
    assert set(result.per_field) == set(grader.FIELD_WEIGHTS)


@pytest.mark.parametrize("submission", [None, "a record", ["invoice"], 42, 1.5, True])
def test_non_dict_submission_scores_zero_with_every_field_missing(submission: Any) -> None:
    result = grader.score(submission, GOLD)
    assert result.total == 0.0
    assert result.per_field == {name: 0.0 for name in grader.FIELD_WEIGHTS}
    assert result.notes == {name: "missing" for name in grader.FIELD_WEIGHTS}


def test_missing_field_is_noted() -> None:
    result = grader.score({k: v for k, v in GOLD.items() if k != "currency"}, GOLD)
    assert result.notes["currency"] == "missing"
    assert result.per_field["currency"] == 0.0


def test_match_line_items_partial_credit() -> None:
    gold_items = GOLD["line_items"]
    assert grader.match_line_items(gold_items, gold_items) == 1.0
    assert grader.match_line_items(gold_items[:2], gold_items) == 0.5
    assert grader.match_line_items(None, gold_items) == 0.0
    assert grader.match_line_items([], []) == 1.0
    duplicated = [gold_items[0], gold_items[0]]
    assert grader.match_line_items(duplicated, gold_items[:1]) == 0.5


# -- tools ----------------------------------------------------------------------------------


def test_registry_loads_by_path_with_six_tools() -> None:
    registry = Registry.load(
        TASK_DIR / "tools.py",
        required=["fetch_document", "submit_record"],
        mixes={},
        rung1=["fetch_document"],
    )
    assert sorted(registry.specs) == sorted(task_tools.REGISTRY)
    assert len(registry.specs) == 6


CALCULATOR_CASES = [
    ("(12.5 * 3) + 4", "41.5"),
    ("1/3", "0.3333333333333333333333333333"),
    ("2 - -3", "5"),
    ("-(2 + 3) * 2", "-10"),
    ("10 / 4", "2.5"),
    ("0.1 + 0.2", "0.3"),
    ("2.50 * 2", "5.00"),
    ("0 * -1", "0"),
    (" 3 + 4 * 2 ", "11"),
    ("(1 + 2) * (3 + 4)", "21"),
    ("8 / 2 / 2", "2"),
    ("10 - 2 - 3", "5"),
    ("+7", "7"),
    (".5 + 1.", "1.5"),
    ("(" * 20 + "1" + ")" * 20, "1"),
    ("99999999999999999999 * 99999999999999999999", "9999999999999999999800000000000000000000"),
]


@pytest.mark.parametrize(("expression", "expected"), CALCULATOR_CASES)
async def test_calculator_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert await task_tools.calculator({"expression": expression}, _ctx("inv-0001")) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "2 ** 3",
        "2 ^ 3",
        "2 % 3",
        "__import__('os').system('echo hi')",
        "1e5",
        "abs(-1)",
        "1 +",
        "(1 + 2",
        "1 + 2)",
        "()",
        "1,000 + 1",
        "1 2",
        "0x10",
        "1.2.3",
        "nan",
        "\u0663 + 1",  # a non-ASCII digit
        "(" * 60 + "1" + ")" * 60,  # deeper than the limit
        "1+" * 600 + "1",  # longer than the limit
    ],
)
async def test_calculator_rejects_everything_else(expression: str) -> None:
    reply = await task_tools.calculator({"expression": expression}, _ctx("inv-0001"))
    assert reply == task_tools.CALCULATOR_PARSE_ERROR
    assert json.loads(reply)["error"] == "invalid_expression"


@pytest.mark.parametrize("expression", ["1 / 0", "0 / 0", "1 / (2 - 2)"])
async def test_calculator_arithmetic_errors_are_frozen(expression: str) -> None:
    reply = await task_tools.calculator({"expression": expression}, _ctx("inv-0001"))
    assert reply == task_tools.CALCULATOR_ARITHMETIC_ERROR


async def test_calculator_never_uses_eval_or_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("eval or exec called")

    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    assert await task_tools.calculator({"expression": "(12.5 * 3) + 4"}, _ctx("inv-0001")) == "41.5"


async def test_search_web_is_a_deterministic_canned_lookup() -> None:
    index = json.loads((DATASET_DIR / dataset.SEARCH_INDEX_NAME).read_text(encoding="utf-8"))
    selections = set()
    for doc_id in ("inv-0001", "inv-0042", "inv-0100"):
        hits = index[doc_id]
        for query in ("invoice total", "vendor address", "", "Quillhaven", "\u00fcber invoice"):
            first = await task_tools.search_web({"query": query}, _ctx(doc_id))
            again = await task_tools.search_web({"query": query}, _ctx(doc_id))
            assert first == again
            reply = json.loads(first)
            start = int(rng.unit("search", doc_id, query) * len(hits))
            expected = [hits[(start + k) % len(hits)] for k in range(3)]
            assert reply == {"query": query, "results": expected}
            assert all(".example/" in hit["url"] for hit in reply["results"])
            selections.add(json.dumps(reply["results"], sort_keys=True))
    assert len(selections) > 3


async def test_search_web_without_an_index_entry_returns_no_results() -> None:
    reply = await task_tools.search_web({"query": "anything"}, _ctx("inv-9999"))
    assert json.loads(reply) == {"query": "anything", "results": []}


def test_search_index_holds_no_gold_values_beyond_the_vendor() -> None:
    index = json.loads((DATASET_DIR / dataset.SEARCH_INDEX_NAME).read_text(encoding="utf-8"))
    gold = dataset.load_gold(DATASET_DIR)
    assert sorted(index) == sorted(gold)
    for doc_id, record in gold.items():
        text = json.dumps(index[doc_id])
        fields = record.fields
        assert fields["invoice_number"] not in text, doc_id
        assert fields["invoice_date"] not in text, doc_id
        assert f"{Decimal(str(fields['total_amount'])):.2f}" not in text, doc_id
        assert fields["vendor_name"] in text
        assert re.search(r"To query invoice \S+,", text), "the hit with a different number"


async def test_read_spreadsheet_renders_the_line_items_as_csv() -> None:
    gold = dataset.load_gold(DATASET_DIR)
    for doc_id in ("inv-0001", "inv-0017", "inv-0100"):
        reply = await task_tools.read_spreadsheet({"doc_id": doc_id}, _ctx(doc_id))
        assert reply == await task_tools.read_spreadsheet({"doc_id": doc_id}, _ctx(doc_id))
        assert reply.startswith("description,quantity,unit_price,amount\n")
        rows = list(csv.DictReader(io.StringIO(reply)))
        fields = gold[doc_id].fields
        assert grader.match_line_items(rows, fields["line_items"]) == 1.0
        assert fields["invoice_number"] not in reply and fields["vendor_name"] not in reply


async def test_read_spreadsheet_quotes_minimally_and_keeps_wrapped_descriptions_whole(
    levels: dict[int, Path],
) -> None:
    level3 = levels[3]
    gold = dataset.load_gold(level3)
    wrapped = [
        doc.id
        for doc in dataset.load(level3)
        if any(
            len(i["description"]) > dataset.WRAP_WIDTH for i in gold[doc.id].fields["line_items"]
        )
    ]
    assert wrapped
    reply = await task_tools.read_spreadsheet({"doc_id": wrapped[0]}, _ctx(wrapped[0], level3))
    descriptions = [row["description"] for row in csv.DictReader(io.StringIO(reply))]
    assert descriptions == [i["description"] for i in gold[wrapped[0]].fields["line_items"]]
    assert '"' in reply  # descriptions with commas are quoted


async def test_read_spreadsheet_not_found_matches_fetch_document() -> None:
    ctx = _ctx("inv-0001")
    for other in ("inv-0002", "", "INV-0001"):
        expected = await task_tools.fetch_document({"doc_id": other}, ctx)
        assert await task_tools.read_spreadsheet({"doc_id": other}, ctx) == expected
        assert json.loads(expected)["error"] == "not_found"


def _every_tool_call(doc_id: str) -> list[tuple[str, dict[str, Any]]]:
    record = dataset.load_gold(DATASET_DIR)[doc_id].fields
    return [
        ("fetch_document", {"doc_id": doc_id}),
        ("submit_record", {"record": record}),
        ("search_web", {"query": "invoice"}),
        ("calculator", {"expression": "(12.5 * 3) + 4"}),
        ("read_spreadsheet", {"doc_id": doc_id}),
        ("send_email", {"to": "accounts@example.com", "subject": "Invoice", "body": "Hello"}),
    ]


async def test_no_tool_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted")

    for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname", "socketpair"):
        monkeypatch.setattr(socket, name, no_network)
    ctx = _ctx("inv-0001")
    calls = _every_tool_call("inv-0001")
    assert sorted(name for name, _ in calls) == sorted(task_tools.REGISTRY)
    for name, args in calls:
        reply = await task_tools.REGISTRY[name].execute(args, ctx)
        assert isinstance(reply, str) and reply, name


async def test_tools_never_need_the_gold_file(tmp_path: Path) -> None:
    """Copy the dataset without gold.jsonl; every tool still answers, and identically."""
    copy = tmp_path / "no_gold"
    shutil.copytree(DATASET_DIR, copy, ignore=shutil.ignore_patterns(dataset.GOLD_NAME))
    assert not (copy / dataset.GOLD_NAME).exists()
    for name, args in _every_tool_call("inv-0005"):
        with_gold = await task_tools.REGISTRY[name].execute(args, _ctx("inv-0005"))
        without = await task_tools.REGISTRY[name].execute(args, _ctx("inv-0005", copy))
        assert with_gold == without, name
