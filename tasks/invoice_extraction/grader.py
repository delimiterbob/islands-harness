"""Executable grader for the public invoice-extraction task.

Returns a continuous score in [0, 1] from weighted field checks against the gold record,
plus per-field detail so a reviewer can see why a run scored what it did. The success cut
(score >= tau) is applied elsewhere (runner.grade_run) from the frozen pre-registration; this
module never sees tau.

This file is hashed by ``islands freeze``. The normalization rules below are the frozen
rules of snapshot 1 (ARCHITECTURE.md Section 10):

Field weights (sum 10):
    invoice_number 2, invoice_date 1, vendor_name 1, currency 1, total_amount 2, line_items 3.

Normalization, applied to both the submission and the gold before comparison:
    strings   case-folded (str.casefold), Unicode NFKC, whitespace collapsed to one space,
              leading and trailing whitespace stripped, trailing punctuation (.,;:) stripped.
              An empty string never matches.
    dates     the string rule above first, then parsed to ISO YYYY-MM-DD from: YYYY-MM-DD,
              DD/MM/YYYY, MM/DD/YYYY (only when unambiguous, day > 12), DD.MM.YYYY,
              D Month YYYY, Month D, YYYY, and YYYYMMDD; anything else is a mismatch. Day
              and month may have one or two digits except in YYYYMMDD; month names are the
              full English names in any case. The date must exist on the calendar.
              Two-digit years are a mismatch.
    amounts   parsed to Decimal from a string or number: currency symbols and codes stripped,
              thousands separators removed (',' or ' ' or '.' when a later ',' is the decimal
              mark), parentheses read as negative; matched when |a - b| <= NUMERIC_TOLERANCE.
              When both ',' and '.' appear, the later one is the decimal mark. A lone ','
              is the decimal mark only when it appears once and is followed by one or two
              digits (12,50 is 12.50; 1,234 is 1234). A '.' without a later ',' is always
              the decimal mark. Booleans, NaN and infinities are not amounts.
    currency  upper-cased three-letter code; symbols are mapped ($ -> USD, EUR sign -> EUR,
              GBP sign -> GBP) only when the gold uses the same currency. The map is applied
              to both sides and the results must be equal, so '$' earns credit only against
              a USD gold. Anything that is not a three-letter code or a mapped symbol is a
              mismatch.

Line items: partial credit. Each gold item is matched to at most one submitted item by
greedy best match on normalized description similarity (exact after normalization first,
then token Jaccard >= 0.6), and a matched item scores the fraction of its four fields that
match (description counted as matched by construction; quantity, unit_price and amount within
NUMERIC_TOLERANCE). The line_items score is (sum of matched item scores) / max(len(gold),
len(submitted)), so extra invented items cost credit.

    exact pass    gold items in order, each takes the first unused submitted item whose
                  normalized description equals its own.
    Jaccard pass  every remaining (gold, submitted) pair with Jaccard >= 0.6 is ranked by
                  Jaccard descending, then gold index, then submitted index, and taken in
                  that order when both sides are still free. Tokens are the runs of word
                  characters (regex \\w+) in the normalized description.
    A submitted entry that is not an object, or has no usable description, never matches
    but still counts in len(submitted). Both lists empty scores 1.

Missing or non-dict submissions score 0 with per_field marking every field missing.
"""

# No ``from __future__ import annotations`` here on purpose: the harness loads this file by
# path without registering it in sys.modules, and dataclasses cannot resolve string
# annotations for an unregistered module. Every annotation below is valid at runtime.

import datetime
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any

FIELD_WEIGHTS: dict[str, int] = {
    "invoice_number": 2,
    "invoice_date": 1,
    "vendor_name": 1,
    "currency": 1,
    "total_amount": 2,
    "line_items": 3,
}

NUMERIC_TOLERANCE: Decimal = Decimal("0.005")

JACCARD_THRESHOLD: Fraction = Fraction(3, 5)

CURRENCY_SYMBOLS: dict[str, str] = {"$": "USD", "\u20ac": "EUR", "\u00a3": "GBP"}

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+")
_CURRENCY_CODE = re.compile(r"[A-Z]{3}")

_MONTHS: dict[str, int] = {
    name: number
    for number, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}

# The date ladder, tried in this order. Each pattern must match the whole normalized string.
_DATE_ISO = re.compile(r"([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})")
_DATE_SLASH = re.compile(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})")
_DATE_DOTTED = re.compile(r"([0-9]{1,2})\.([0-9]{1,2})\.([0-9]{4})")
_DATE_DAY_MONTH = re.compile(r"([0-9]{1,2}) ([a-z]+) ([0-9]{4})")
_DATE_MONTH_DAY = re.compile(r"([a-z]+) ([0-9]{1,2}), ([0-9]{4})")
_DATE_COMPACT = re.compile(r"([0-9]{4})([0-9]{2})([0-9]{2})")

_EDGE_CURRENCY_CODE = re.compile(r"^[a-z]{3}|[a-z]{3}$")
_AMOUNT_CHARS = re.compile(r"[0-9.,]+")
_PLAIN_DECIMAL = re.compile(r"[0-9]+(\.[0-9]+)?")


@dataclass(frozen=True)
class GradeResult:
    """The continuous score and its decomposition.

    total: weighted score in [0, 1].
    per_field: field name -> credit in [0, 1] for that field (before weighting).
    notes: field name -> short reason string for a reviewer (never read by the statistics).
    """

    total: float
    per_field: dict[str, float]
    notes: dict[str, str] = field(default_factory=dict)


def normalize_string(value: Any) -> str:
    """Casefold, NFKC-normalize, collapse whitespace, strip edges and trailing punctuation.

    Non-strings are rendered with str() first; None becomes the empty string.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = _WS.sub(" ", text).strip()
    return text.rstrip(".,;:").strip()


def _calendar_date(year: str, month: str, day: str) -> str | None:
    """ISO string for the given parts, or None when the date does not exist."""
    try:
        return datetime.date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def normalize_date(value: Any) -> str | None:
    """Parse the accepted date formats to ISO YYYY-MM-DD; None when not parseable.

    Accepted formats are listed in the module docstring and are frozen. Ambiguous numeric
    dates (both parts <= 12 in DD/MM and MM/DD) resolve to DD/MM/YYYY because the generator
    writes day-first in that layout; the gold is always ISO so the rule only affects
    submissions.
    """
    text = normalize_string(value)
    if match := _DATE_ISO.fullmatch(text):
        year, month, day = match.groups()
        return _calendar_date(year, month, day)
    if match := _DATE_SLASH.fullmatch(text):
        first, second, year = match.groups()
        if int(first) <= 12 < int(second):
            return _calendar_date(year, first, second)  # MM/DD/YYYY, day > 12
        return _calendar_date(year, second, first)  # DD/MM/YYYY, including ambiguous
    if match := _DATE_DOTTED.fullmatch(text):
        day, month, year = match.groups()
        return _calendar_date(year, month, day)
    if match := _DATE_DAY_MONTH.fullmatch(text):
        day, month_name, year = match.groups()
        if month_name not in _MONTHS:
            return None
        return _calendar_date(year, str(_MONTHS[month_name]), day)
    if match := _DATE_MONTH_DAY.fullmatch(text):
        month_name, day, year = match.groups()
        if month_name not in _MONTHS:
            return None
        return _calendar_date(year, str(_MONTHS[month_name]), day)
    if match := _DATE_COMPACT.fullmatch(text):
        year, month, day = match.groups()
        return _calendar_date(year, month, day)
    return None


def _decimal_text(text: str) -> str | None:
    """Digits and separators (spaces already removed) to a plain decimal string, or None.

    Applies the separator rule of the module docstring.
    """
    if not _AMOUNT_CHARS.fullmatch(text):
        return None
    has_comma = "," in text
    has_dot = "." in text
    if has_comma and has_dot:
        decimal_mark = "," if text.rindex(",") > text.rindex(".") else "."
    elif has_comma:
        digits_after = len(text) - text.rindex(",") - 1
        lone_decimal = text.count(",") == 1 and digits_after in (1, 2)
        decimal_mark = "," if lone_decimal else ""
    else:
        decimal_mark = "."
    thousands_mark = "." if decimal_mark == "," else ","
    text = text.replace(thousands_mark, "")
    if decimal_mark == ",":
        text = text.replace(",", ".")
    if not _PLAIN_DECIMAL.fullmatch(text):
        return None
    return text


def _parse_amount_string(value: str) -> Decimal | None:
    """The string branch of normalize_amount."""
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    for symbol in CURRENCY_SYMBOLS:
        text = text.replace(symbol, "")
    text = _EDGE_CURRENCY_CODE.sub("", text.strip()).strip()
    if text.startswith("-"):
        if negative:
            return None
        negative = True
        text = text[1:].strip()
    plain = _decimal_text(text.replace(" ", ""))
    if plain is None:
        return None
    amount = Decimal(plain)
    return -amount if negative else amount


def normalize_amount(value: Any) -> Decimal | None:
    """Parse a number or an amount string to Decimal; None when not parseable.

    Strips currency symbols and codes, thousands separators, and reads parentheses as
    negative, per the module docstring. Floats are converted through str() to avoid binary
    artefacts (12.30 -> Decimal("12.3")).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        amount = Decimal(str(value))
        return amount if amount.is_finite() else None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, str):
        return _parse_amount_string(value)
    return None


def normalize_currency(value: Any) -> str:
    """Upper-cased three-letter code, or the symbol mapping in the module docstring.

    Returns the empty string for anything else, which never matches.
    """
    text = normalize_string(value)
    if text in CURRENCY_SYMBOLS:
        return CURRENCY_SYMBOLS[text]
    text = text.upper()
    return text if _CURRENCY_CODE.fullmatch(text) else ""


def _amounts_match(submitted: Any, gold: Any) -> bool:
    a = normalize_amount(submitted)
    b = normalize_amount(gold)
    return a is not None and b is not None and abs(a - b) <= NUMERIC_TOLERANCE


def _jaccard(a: str, b: str) -> Fraction:
    tokens_a = set(_TOKEN.findall(a))
    tokens_b = set(_TOKEN.findall(b))
    union = tokens_a | tokens_b
    if not union:
        return Fraction(0)
    return Fraction(len(tokens_a & tokens_b), len(union))


def _item_description(item: Any) -> str:
    """Normalized description of a submitted item; empty when it cannot be matched."""
    if not isinstance(item, dict):
        return ""
    return normalize_string(item.get("description"))


def _pair_items(submitted: list[Any], gold: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(gold index, submitted index) pairs from the exact pass, then the Jaccard pass."""
    gold_descriptions = [normalize_string(item.get("description")) for item in gold]
    sub_descriptions = [_item_description(item) for item in submitted]
    pairs: list[tuple[int, int]] = []
    used_gold: set[int] = set()
    used_sub: set[int] = set()
    for g, gold_text in enumerate(gold_descriptions):
        for s, sub_text in enumerate(sub_descriptions):
            if s not in used_sub and sub_text and sub_text == gold_text:
                pairs.append((g, s))
                used_gold.add(g)
                used_sub.add(s)
                break
    candidates: list[tuple[Fraction, int, int]] = []
    for g, gold_text in enumerate(gold_descriptions):
        for s, sub_text in enumerate(sub_descriptions):
            if g in used_gold or s in used_sub or not sub_text:
                continue
            similarity = _jaccard(gold_text, sub_text)
            if similarity >= JACCARD_THRESHOLD:
                candidates.append((-similarity, g, s))
    for _, g, s in sorted(candidates):
        if g not in used_gold and s not in used_sub:
            pairs.append((g, s))
            used_gold.add(g)
            used_sub.add(s)
    return pairs


def _item_score(submitted: dict[str, Any], gold: dict[str, Any]) -> float:
    """Fraction of the four fields that match; the description counts by construction."""
    matched = 1
    for key in ("quantity", "unit_price", "amount"):
        if _amounts_match(submitted.get(key), gold.get(key)):
            matched += 1
    return matched / 4


def _line_items_detail(submitted: Any, gold: list[dict[str, Any]]) -> tuple[float, str]:
    """The line_items credit and a short note for a reviewer."""
    if not isinstance(submitted, list):
        return 0.0, "not a list"
    denominator = max(len(gold), len(submitted))
    if denominator == 0:
        return 1.0, "both empty"
    pairs = _pair_items(submitted, gold)
    earned = sum(_item_score(submitted[s], gold[g]) for g, s in pairs)
    note = f"{len(pairs)} of {len(gold)} gold items matched; {len(submitted)} submitted"
    return earned / denominator, note


def match_line_items(submitted: Any, gold: list[dict[str, Any]]) -> float:
    """Partial credit for line items in [0, 1] per the rule in the module docstring.

    Greedy matching by normalized description (exact, then token Jaccard >= 0.6), each matched
    item scoring the fraction of its four fields that match, divided by
    max(len(gold), len(submitted)). A non-list submission scores 0.
    """
    credit, _ = _line_items_detail(submitted, gold)
    return credit


def _scalar_credit(name: str, submitted: Any, gold: Any) -> bool:
    """1-or-0 check for the five scalar fields after their normalization."""
    if name == "invoice_date":
        sub_date = normalize_date(submitted)
        return sub_date is not None and sub_date == normalize_date(gold)
    if name == "currency":
        sub_code = normalize_currency(submitted)
        return sub_code != "" and sub_code == normalize_currency(gold)
    if name == "total_amount":
        return _amounts_match(submitted, gold)
    sub_text = normalize_string(submitted)
    return sub_text != "" and sub_text == normalize_string(gold)


def score(submission: Any, gold: dict[str, Any]) -> GradeResult:
    """Score a submitted record against the gold record.

    total = sum(weight_f * credit_f) / sum(weight_f) over FIELD_WEIGHTS, where credit_f is 1
    or 0 for scalar fields (after normalization) and match_line_items() for line_items.
    A submission that is not a dict scores 0 on every field.
    """
    if not isinstance(submission, dict):
        per_field = {name: 0.0 for name in FIELD_WEIGHTS}
        notes = {name: "missing" for name in FIELD_WEIGHTS}
        return GradeResult(total=0.0, per_field=per_field, notes=notes)
    per_field: dict[str, float] = {}
    notes: dict[str, str] = {}
    for name in FIELD_WEIGHTS:
        if name not in submission:
            per_field[name] = 0.0
            notes[name] = "missing"
        elif name == "line_items":
            per_field[name], notes[name] = _line_items_detail(
                submission[name], gold.get("line_items", [])
            )
        else:
            matched = _scalar_credit(name, submission[name], gold.get(name))
            per_field[name] = 1.0 if matched else 0.0
            notes[name] = "match" if matched else "mismatch"
    weighted = sum(FIELD_WEIGHTS[name] * per_field[name] for name in FIELD_WEIGHTS)
    total = weighted / sum(FIELD_WEIGHTS.values())
    return GradeResult(total=total, per_field=per_field, notes=notes)
