"""Synthetic invoice generator, loader and lock verification.

Generation is deterministic given ``(seed, n, difficulty)``: every draw is
``rng.unit("dataset", seed, difficulty, i, field)`` so the same call reproduces the same
documents and gold records byte for byte on any platform. The dataset is written once,
locked by per-file hash, and verified before every run; a regenerated dataset with any
difference is a different snapshot.

Draws: ``i`` is the 1-based document number (``inv-0007`` is i = 7) and ``field`` is a string
naming the draw, for example ``"date.month"`` or ``"item.3.quantity"``. An integer in
[lo, hi] is ``lo + floor(u * (hi - lo + 1))``; a choice maps ``u`` through
``rng.choice_weighted`` with equal weights; an ordering sorts by one draw per element. There
is no other source of randomness here: no ``random`` module, no clock.

Layout under ``out_dir``:

    documents/<doc_id>.txt     the rendered invoice text the agent fetches
    gold.jsonl                 one {"doc_id": ..., "record": {...}} per line, sorted by doc_id
    search_index.json          canned search_web hits per document (never contains gold values
                               beyond the vendor name, which is printed in the document anyway)
    spreadsheets.json          per document, the generated line items as the plain strings
                               read_spreadsheet renders to CSV (no header fields, no labels)
    LOCK.json                  {"schema_version": 1, "seed", "n", "difficulty",
                               "files": {path: sha256}}

Document ids are ``inv-0001`` .. ``inv-nnnn`` (zero-padded to four digits).

Difficulty knobs (level 1 to 3; each level adds to the previous):

    1  clean layout: labelled fields in a fixed order, ISO dates, one currency (USD),
       2 to 4 line items, no distractors.
    2  layout noise: field order varies per document, one of four date formats, three
       currencies with symbols, 2 to 6 line items, one distractor block (a "previous balance"
       or a "PO number" line that looks like an invoice number), subtotal and tax lines.
    3  harder: two distractor blocks including a quoted earlier invoice number, amounts with
       thousands separators and mixed decimal marks by currency, up to 8 line items with
       discounts, a shipping line that is not a line item, occasional wrapped descriptions.
    4  computed amounts: everything in level 3, but no line amount, subtotal, tax amount or
       total is printed. The document states the rules instead (each line is quantity times
       unit price less its discount, rounded half up to the cent; tax at the printed rate on
       the sum of the lines, rounded the same way; shipping is not taxed), so the record's
       amounts must be computed. Added on 2026-09-26 after calibration found two models at
       100 percent on levels 1 to 3. read_spreadsheet still returns each line's amount and
       the calculator can do the arithmetic, so from 4 tools up a tool can do what the
       reader otherwise must.

How each knob is rendered:

    header       the vendor (its name, then its address on an indented line), the invoice
                 number, the invoice date, the currency code and the bill-to customer (name,
                 then address), one "Label: value" line each. Level 1 uses the first label of
                 each LABELS entry in the order of HEADER_BLOCKS; levels 2 and 3 draw a label
                 per field and shuffle the header blocks, header distractors included.
    dates        level 1 ISO (2026-03-14); levels 2 and 3 one of ISO, 14/03/2026, 14 March
                 2026 and March 14, 2026. The day-first slash form is drawn only with days 13
                 to 28, so every printed date has exactly one reading. Days run 1 to 28.
    amounts      level 1 plain (1234.50) with no symbol; level 2 plain in the table and with
                 a leading symbol in the totals block; level 3 grouped, USD and GBP as
                 1,234.50 and EUR as 1.234,50, with the EUR sign after the number in the
                 totals block and the other symbols before it.
    table        fixed-width columns separated by at least two spaces, between two rules of
                 dashes: Description, Qty, Unit price, [Disc.,] Amount. Level 3 adds the
                 discount column (blank or a percentage; unit price before discount, amount
                 after it, rounded half up to the cent) and wraps descriptions longer than
                 WRAP_WIDTH at word boundaries onto continuation lines indented two spaces.
    totals       level 1 a single total; level 2 subtotal, tax at a rate drawn per currency,
                 total; level 3 adds a shipping line between subtotal and tax. Tax is charged
                 on the subtotal only, and total = subtotal + shipping + tax.
    distractors  "previous balance" is a footer line with an amount and a received-on date;
                 "PO number" and the level 3 "earlier invoice" note are header lines whose
                 numbers have the invoice number's shape and never equal it.

The grader's normalization rules (tasks/invoice_extraction/grader.py) are designed so that
a correct reading of any level scores 1.0; difficulty changes how easy the reading is, not
what counts as correct.
"""

from __future__ import annotations

import datetime
import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from islands_harness import rng
from islands_harness.config import LockMismatch, sha256_file

DATASET_LOCK_NAME = "LOCK.json"
DOCUMENTS_DIR = "documents"
GOLD_NAME = "gold.jsonl"
SEARCH_INDEX_NAME = "search_index.json"
SPREADSHEETS_NAME = "spreadsheets.json"
SPREADSHEET_COLUMNS: tuple[str, ...] = ("description", "quantity", "unit_price", "amount")

DIFFICULTY_LEVELS: tuple[int, ...] = (1, 2, 3, 4)
MAX_DOCUMENTS = 9999
WRAP_WIDTH = 34
CENT = Decimal("0.01")

# --------------------------------------------------------------------------------------
# Word lists. Invented names: any resemblance to a real business, person or place is
# coincidental, and every web address uses the reserved .example domain.
# --------------------------------------------------------------------------------------

VENDOR_STEMS: tuple[str, ...] = (
    "Quillhaven",
    "Mossbeck",
    "Tinderwick",
    "Hollowmere",
    "Pebblequay",
    "Frostvane",
    "Emberlyn",
    "Kestrelmoor",
    "Duskwater",
    "Wintercress",
    "Amberquill",
    "Nettlewhistle",
    "Glimmerholt",
    "Saltreed",
    "Harrowfinch",
    "Marrowdell",
    "Oakenquist",
    "Thistlewren",
    "Cobblemere",
    "Larkvane",
    "Fenwhistle",
    "Brindlequay",
    "Starlingholt",
    "Copperwick",
)

CUSTOMER_STEMS: tuple[str, ...] = (
    "Ashquill",
    "Bellmarrow",
    "Cindervale",
    "Dewhollow",
    "Elderwick",
    "Foxglade",
    "Gorsemere",
    "Hazelquay",
    "Ivywhistle",
    "Juniperholt",
    "Kilnbrook",
    "Lindenquist",
)

CUSTOMER_KINDS: tuple[str, ...] = (
    "Dental Studio",
    "Community Library",
    "Architects",
    "Veterinary Clinic",
    "Bakery",
    "Primary School",
    "Rowing Club",
    "Physiotherapy",
    "Theatre Company",
    "Cycle Works",
)

STREETS: tuple[str, ...] = (
    "Larkspur",
    "Thimble",
    "Wicket",
    "Pennyroyal",
    "Saltmarsh",
    "Cobble",
    "Heron",
    "Juniper",
    "Kettle",
    "Lantern",
    "Marigold",
    "Orchard",
)

STREET_TYPES: tuple[str, ...] = ("Lane", "Row", "Street", "Way", "Court", "Road")

CITIES: tuple[str, ...] = (
    "Fernhollow",
    "Glassmere Vale",
    "Quillinby",
    "Harrowvane",
    "Ottersnook",
    "Brambleketh",
    "Wetherquill",
    "Mossmarrow",
    "Pellwick Cross",
    "Starling Hythe",
    "Duskmere",
    "Lindenharrow",
)

CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP")

CURRENCY_SYMBOLS: dict[str, str] = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3"}

LEGAL_SUFFIXES: dict[str, tuple[str, ...]] = {
    "USD": ("Inc.", "LLC", "Co."),
    "EUR": ("GmbH", "B.V.", "S.A."),
    "GBP": ("Ltd", "Ltd.", "& Co. Ltd"),
}

# Tax name and the rates (percent) drawn from, per currency.
TAX_RULES: dict[str, tuple[str, tuple[str, ...]]] = {
    "USD": ("Sales tax", ("6.5", "7.25", "8", "8.875")),
    "EUR": ("VAT", ("7", "19", "21")),
    "GBP": ("VAT", ("5", "20")),
}

# Products per trade: (description, typical unit price in whole units, maximum quantity).
# A document's unit price is drawn between 70 and 130 percent of the typical price.
TRADES: dict[str, tuple[tuple[str, int, int], ...]] = {
    "Office Supplies": (
        ("A4 copy paper, 80 gsm, ream of 500", 6, 40),
        ("Ballpoint pens, blue, box of 50", 12, 20),
        ("Lever arch files, assorted colours, pack of 10", 18, 15),
        ("Sticky notes 76 x 76 mm, pack of 12", 9, 30),
        ("Desk organiser, recycled plastic", 14, 10),
        ("Heavy duty stapler, full strip", 24, 6),
        ("Whiteboard markers, set of 8", 7, 25),
        ("Laminating pouches A4, pack of 100", 15, 12),
        ("Archive storage boxes, flat packed, pack of 10", 21, 10),
        ("Self seal envelopes C5, box of 500", 19, 10),
    ),
    "Packaging": (
        ("Kraft mailer boxes, medium, pack of 25", 22, 20),
        ("Bubble wrap roll 750 mm x 100 m", 31, 10),
        ("Clear packing tape 48 mm, 36 rolls", 38, 8),
        ("Stretch film 500 mm, cast, 6 rolls", 45, 6),
        ("Void fill paper roll, 5 kg", 27, 12),
        ("Fragile labels, roll of 1000", 11, 15),
        ("Corrugated sheets 1200 x 800 mm, pack of 50", 64, 5),
        ("Poly mailing bags, large, pack of 500", 36, 8),
        ("Pallet corner protectors, pack of 200", 42, 6),
        ("Tamper evident seals, roll of 500", 17, 10),
    ),
    "Catering Supplies": (
        ("Arabica coffee beans, 1 kg bag", 19, 20),
        ("Paper cups 8 oz, sleeve of 1000", 34, 10),
        ("Loose leaf breakfast tea, 500 g", 11, 15),
        ("Compostable cutlery sets, box of 250", 29, 8),
        ("Oat milk, barista edition, case of 6", 13, 20),
        ("Napkins, two ply, white, pack of 500", 8, 25),
        ("Sugar sticks, box of 1000", 12, 10),
        ("Stainless steel water jug, 1.5 litre", 16, 6),
        ("Food storage containers with lids, set of 20", 26, 6),
        ("Hand soap refill, 5 litre", 15, 10),
    ),
    "IT Services": (
        ("On-site support, per hour", 95, 16),
        ("Laptop setup and imaging", 120, 10),
        ("Network cabling survey, half day visit", 340, 3),
        ("Monthly backup monitoring", 180, 3),
        ("Firewall configuration review", 450, 2),
        ("Help desk retainer, monthly, up to 20 tickets", 600, 2),
        ("Data migration, per hour", 110, 12),
        ("Printer maintenance visit", 85, 4),
        ("Wireless access point installation", 160, 6),
        ("Password manager licence, per user per year", 36, 40),
    ),
    "Grounds Care": (
        ("Lawn mowing and edging, per visit", 65, 8),
        ("Hedge trimming, per linear metre", 4, 60),
        ("Bark mulch, bulk bag of 1 cubic metre", 78, 6),
        ("Leaf clearance, per hour", 38, 12),
        ("Winter gritting service, per callout", 140, 5),
        ("Planter refresh with seasonal bulbs", 55, 10),
        ("Tree survey and written report", 390, 1),
        ("Weed control treatment, per visit", 72, 6),
        ("Gutter clearing, per building", 120, 4),
        ("Green waste removal, per load", 95, 5),
    ),
    "Lab Supplies": (
        ("Nitrile gloves, medium, box of 100", 9, 30),
        ("Pipette tips 200 ul, rack of 96", 14, 25),
        ("Glass beakers 250 ml, pack of 12", 32, 6),
        ("Cotton lab coats, size large", 28, 10),
        ("Sample vials 2 ml with caps, pack of 100", 23, 12),
        ("Distilled water, 5 litre container", 7, 20),
        ("pH indicator strips, pack of 100", 12, 15),
        ("Anti fog safety goggles", 11, 20),
        ("Microscope slides, frosted end, box of 72", 15, 10),
        ("Centrifuge tubes 50 ml, sterile, bag of 25", 19, 12),
    ),
    "Print Studio": (
        ("Business cards, matt, box of 500", 45, 6),
        ("A5 flyers, double sided, 2000 copies", 120, 3),
        ("Roller banner 850 x 2000 mm with stand", 95, 4),
        ("A2 posters, gloss, pack of 50", 68, 5),
        ("Presentation folders, box of 250", 210, 2),
        ("Letterheads 100 gsm, box of 1000", 85, 4),
        ("Round vinyl stickers 60 mm, roll of 500", 48, 6),
        ("Wall calendars, spiral bound, 100 copies", 330, 2),
        ("Design proof and artwork check, per hour", 55, 6),
        ("Saddle stitched booklets A5, 250 copies", 290, 2),
    ),
    "Workshop Tools": (
        ("Cordless drill driver 18 V, body only", 89, 5),
        ("Metric hex key set, 9 piece", 12, 12),
        ("Safety boots, steel toe, size 42", 64, 8),
        ("Cut resistant work gloves, 12 pairs", 38, 10),
        ("Tape measure 8 m, magnetic hook", 14, 15),
        ("Spirit level 600 mm, aluminium", 22, 8),
        ("HSS drill bit set, 19 piece", 26, 10),
        ("Cable ties 300 mm, bag of 100", 6, 30),
        ("Adjustable spanner 250 mm", 17, 10),
        ("Rechargeable work light, 20 W", 45, 6),
    ),
}

ITEM_COUNTS: dict[int, tuple[int, int]] = {1: (2, 4), 2: (2, 6), 3: (2, 8), 4: (2, 8)}
DISCOUNT_PERCENTS: tuple[int, ...] = (5, 10, 15, 20)
DISCOUNT_CHANCE = 0.3

YEARS: tuple[int, ...] = (2025, 2026)
DATE_STYLES: tuple[str, ...] = ("iso", "day_slash", "day_month", "month_day")
MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# Invoice number shapes and the sequence range each draws from.
NUMBER_SHAPES: dict[str, tuple[int, int]] = {
    "code_year_seq": (1000, 9999),  # QUI-2026-4413
    "seven_digits": (1000000, 9999999),  # 2604471
    "code_six_digits": (100000, 999999),  # QU204417
}

HEADER_BLOCKS: tuple[str, ...] = ("vendor", "number", "date", "currency", "bill_to")

# Label choices per field. Level 1 always uses the first.
LABELS: dict[str, tuple[str, ...]] = {
    "vendor": ("Vendor", "From", "Supplier", "Issued by"),
    "number": ("Invoice number", "Invoice no.", "Invoice #", "Invoice ID"),
    "date": ("Invoice date", "Date of issue", "Issue date", "Invoice dated"),
    "currency": ("Currency", "Billing currency", "Invoice currency"),
    "bill_to": ("Bill to", "Customer", "Invoice to"),
    "total": ("Total", "Total due", "Invoice total", "Amount due"),
    "po_number": ("PO number", "Purchase order", "Your order no."),
    "shipping": ("Shipping and handling", "Delivery", "Freight"),
}

TITLES: tuple[str, ...] = ("INVOICE", "TAX INVOICE", "Invoice")
PAYMENT_TERMS: tuple[str, ...] = ("30 days", "14 days", "45 days")

# Level 2 draws one of these; level 3 always has "earlier_invoice" plus one of these.
DISTRACTORS: tuple[str, ...] = ("previous_balance", "po_number")

# Canned search content that is the same for every document (search_web).
GENERIC_ARTICLES: tuple[dict[str, str], ...] = (
    {
        "title": "A short history of the invoice",
        "url": "https://www.ledgerlark.example/history-of-the-invoice",
        "snippet": "Merchants have written bills of sale for thousands of years; the printed "
        "invoice spread with double-entry bookkeeping.",
    },
    {
        "title": "Choosing an invoice template for a small business",
        "url": "https://www.tallyquill.example/templates",
        "snippet": "Templates differ in layout and typeface. Pick one that matches your brand "
        "and prints cleanly on plain paper.",
    },
    {
        "title": "How long should a business keep its invoices?",
        "url": "https://www.sumwhistle.example/record-keeping",
        "snippet": "Record keeping rules vary between jurisdictions; many businesses keep "
        "invoices for several years.",
    },
)
DATE_ARTICLE: dict[str, str] = {
    "title": "Writing dates in ISO 8601 format",
    "url": "https://www.calendrift.example/iso-8601",
    "snippet": "ISO 8601 writes a calendar date as year, month and day separated by hyphens, "
    "for example 1999-12-31.",
}
CURRENCY_ARTICLE: dict[str, str] = {
    "title": "Currency codes and symbols",
    "url": "https://www.calendrift.example/currency-codes",
    "snippet": "ISO 4217 gives each currency a three-letter code. Some symbols are shared by "
    "several currencies.",
}

_SWAP_DECIMAL_MARKS = str.maketrans(",.", ".,")


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class GoldRecord:
    doc_id: str
    fields: dict[str, Any]


# --------------------------------------------------------------------------------------
# Draws
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Draws:
    """Every draw for one document: ``rng.unit("dataset", seed, difficulty, i, field)``."""

    seed: int
    difficulty: int
    i: int

    def unit(self, field: str) -> float:
        return rng.unit("dataset", self.seed, self.difficulty, self.i, field)

    def integer(self, field: str, lo: int, hi: int) -> int:
        """An integer in [lo, hi]; u * span never rounds up to span for u < 1."""
        return lo + int(self.unit(field) * (hi - lo + 1))

    def pick(self, field: str, items: Sequence[Any]) -> Any:
        return rng.choice_weighted(self.unit(field), items, [1] * len(items))

    def chance(self, field: str, probability: float) -> bool:
        return self.unit(field) < probability

    def order(self, field: str, items: Sequence[str]) -> list[str]:
        """``items`` sorted by one draw each, keyed by the item itself."""
        return sorted(items, key=lambda item: self.unit(f"{field}.{item}"))


@dataclass(frozen=True)
class _LineItem:
    description: str
    quantity: int
    unit_price: Decimal
    discount: int  # percent; 0 when the item has no discount
    amount: Decimal


@dataclass(frozen=True)
class _Invoice:
    """Every value one document needs, drawn once and then written four ways."""

    doc_id: str
    difficulty: int
    title: str
    labels: dict[str, str]
    header_order: tuple[str, ...]
    vendor_name: str
    vendor_address: str
    customer_name: str
    customer_address: str
    number: str
    date: datetime.date
    date_style: str
    currency: str
    items: tuple[_LineItem, ...]
    subtotal: Decimal
    shipping: Decimal | None
    tax_label: str | None
    tax: Decimal | None
    total: Decimal
    po_number: str | None
    earlier_number: str | None
    earlier_date: datetime.date | None
    previous_balance: Decimal | None
    previous_date: datetime.date | None
    payment_terms: str
    search_hits: list[dict[str, str]]
    tax_name: str | None = None  # level 4 states the rate and computes nothing
    tax_rate: str | None = None


def _money_from_cents(cents: int) -> Decimal:
    return Decimal(cents) * CENT


def _draw_day(draws: _Draws, field: str, style: str) -> int:
    """Days 1 to 28, or 13 to 28 for the day-first slash style so it has one reading."""
    lo = 13 if style == "day_slash" else 1
    return draws.integer(f"{field}.day", lo, 28)


def _draw_date(draws: _Draws, style: str) -> datetime.date:
    year = draws.pick("date.year", YEARS)
    month = draws.integer("date.month", 1, 12)
    return datetime.date(year, month, _draw_day(draws, "date", style))


def _earlier_date(draws: _Draws, field: str, date: datetime.date, style: str) -> datetime.date:
    """A date one to three calendar months before ``date``."""
    months_back = draws.integer(f"{field}.months_back", 1, 3)
    year, month_index = divmod(date.year * 12 + date.month - 1 - months_back, 12)
    return datetime.date(year, month_index + 1, _draw_day(draws, field, style))


def _render_number(shape: str, code: str, year: int, seq: int) -> str:
    if shape == "code_year_seq":
        return f"{code}-{year}-{seq:04d}"
    if shape == "seven_digits":
        return f"{seq:07d}"
    return f"{code[:2]}{seq:06d}"


def _distinct_number(
    draws: _Draws, field: str, shape: str, code: str, year: int, taken: set[str]
) -> str:
    """A number of the given shape that is not in ``taken`` (step the sequence until free)."""
    lo, hi = NUMBER_SHAPES[shape]
    seq = draws.integer(field, lo, hi)
    number = _render_number(shape, code, year, seq)
    while number in taken:
        seq = lo if seq == hi else seq + 1
        number = _render_number(shape, code, year, seq)
    return number


def _draw_address(draws: _Draws, field: str) -> str:
    number = draws.integer(f"{field}.number", 1, 240)
    street = draws.pick(f"{field}.street", STREETS)
    street_type = draws.pick(f"{field}.street_type", STREET_TYPES)
    city = draws.pick(f"{field}.city", CITIES)
    postcode = draws.integer(f"{field}.postcode", 10000, 99999)
    return f"{number} {street} {street_type}, {city} {postcode}"


def _draw_items(draws: _Draws, trade: str) -> tuple[_LineItem, ...]:
    """Distinct products of the trade in drawn order, with quantity, price and discount."""
    lo, hi = ITEM_COUNTS[draws.difficulty]
    count = draws.integer("items.count", lo, hi)
    catalog = {description: (price, max_qty) for description, price, max_qty in TRADES[trade]}
    chosen = draws.order("items.order", tuple(catalog))[:count]
    items: list[_LineItem] = []
    for j, description in enumerate(chosen, start=1):
        price, max_quantity = catalog[description]
        quantity = draws.integer(f"item.{j}.quantity", 1, max_quantity)
        unit_price = _money_from_cents(draws.integer(f"item.{j}.cents", price * 70, price * 130))
        discount = 0
        if draws.difficulty >= 3 and draws.chance(f"item.{j}.discounted", DISCOUNT_CHANCE):
            discount = draws.pick(f"item.{j}.discount", DISCOUNT_PERCENTS)
        net = unit_price * quantity * (100 - discount) / 100
        amount = net.quantize(CENT, rounding=ROUND_HALF_UP)
        items.append(_LineItem(description, quantity, unit_price, discount, amount))
    return tuple(items)


def _search_hits(
    draws: _Draws, vendor_name: str, stem: str, trade: str, other_number: str
) -> list[dict[str, str]]:
    """The document's canned search_web list: two vendor pages and three generic articles.

    The billing page quotes ``other_number``, which is never the invoice number.
    """
    slug = "-".join([stem.lower(), *trade.lower().split()])
    site = f"https://www.{slug}.example"
    return [
        {
            "title": vendor_name,
            "url": f"{site}/",
            "snippet": f"{vendor_name}: catalogue, delivery information and account help "
            f"for {trade} customers.",
        },
        {
            "title": f"Billing questions | {vendor_name}",
            "url": f"{site}/billing",
            "snippet": f"Invoices are emailed at dispatch. To query invoice {other_number}, "
            "contact the accounts team and quote your customer reference.",
        },
        dict(draws.pick("search.article", GENERIC_ARTICLES)),
        dict(DATE_ARTICLE),
        dict(CURRENCY_ARTICLE),
    ]


def _draw_invoice(seed: int, difficulty: int, i: int) -> _Invoice:
    """Draw every value of document ``i``; rendering adds no randomness of its own."""
    draws = _Draws(seed, difficulty, i)
    level1 = difficulty == 1
    currency = "USD" if level1 else draws.pick("currency", CURRENCIES)
    trade = draws.pick("trade", tuple(TRADES))
    stem = draws.pick("vendor.stem", VENDOR_STEMS)
    vendor_name = f"{stem} {trade} {draws.pick('vendor.suffix', LEGAL_SUFFIXES[currency])}"
    customer_stem = draws.pick("customer.stem", CUSTOMER_STEMS)
    customer_name = f"{customer_stem} {draws.pick('customer.kind', CUSTOMER_KINDS)}"
    date_style = "iso" if level1 else draws.pick("date.style", DATE_STYLES)
    date = _draw_date(draws, date_style)

    code = stem[:3].upper()
    shape = draws.pick("number.shape", tuple(NUMBER_SHAPES))
    lo, hi = NUMBER_SHAPES[shape]
    seq = draws.integer("number.seq", lo + 1000, hi)
    number = _render_number(shape, code, date.year, seq)

    distractors: tuple[str, ...] = ()
    if difficulty == 2:
        distractors = (draws.pick("distractor", DISTRACTORS),)
    elif difficulty >= 3:
        distractors = ("earlier_invoice", draws.pick("distractor", DISTRACTORS))

    taken = {number}
    earlier_number = earlier_date = None
    if "earlier_invoice" in distractors:
        earlier_date = _earlier_date(draws, "earlier", date, date_style)
        earlier_seq = seq - draws.integer("earlier.gap", 1, 999)
        earlier_number = _render_number(shape, code, earlier_date.year, earlier_seq)
        taken.add(earlier_number)
    po_number = None
    if "po_number" in distractors:
        po_number = _distinct_number(draws, "po.seq", shape, code, date.year, taken)
        taken.add(po_number)
    other_number = _distinct_number(draws, "search.seq", shape, code, date.year, taken)

    items = _draw_items(draws, trade)
    subtotal = sum((item.amount for item in items), Decimal("0"))
    shipping = None
    if difficulty >= 3:
        shipping = _money_from_cents(draws.integer("shipping.cents", 495, 4995))
    tax_label = tax = tax_name = rate = None
    if not level1:
        tax_name, rates = TAX_RULES[currency]
        rate = draws.pick("tax.rate", rates)
        tax_label = f"{tax_name} ({rate}%)"
        tax = (subtotal * Decimal(rate) / 100).quantize(CENT, rounding=ROUND_HALF_UP)
    total = subtotal + (shipping or Decimal("0")) + (tax or Decimal("0"))

    previous_balance = previous_date = None
    if "previous_balance" in distractors:
        previous_balance = _money_from_cents(draws.integer("previous.cents", 5000, 300000))
        if previous_balance == total:
            previous_balance += Decimal("1.00")
        previous_date = _earlier_date(draws, "previous", date, date_style)

    if level1:
        labels = {name: choices[0] for name, choices in LABELS.items()}
        header_order = HEADER_BLOCKS
    else:
        labels = {name: draws.pick(f"label.{name}", choices) for name, choices in LABELS.items()}
        header_blocks = HEADER_BLOCKS + tuple(d for d in distractors if d != "previous_balance")
        header_order = tuple(draws.order("header", header_blocks))

    return _Invoice(
        doc_id=f"inv-{i:04d}",
        difficulty=difficulty,
        title=TITLES[0] if level1 else draws.pick("title", TITLES),
        labels=labels,
        header_order=header_order,
        vendor_name=vendor_name,
        vendor_address=_draw_address(draws, "vendor.address"),
        customer_name=customer_name,
        customer_address=_draw_address(draws, "customer.address"),
        number=number,
        date=date,
        date_style=date_style,
        currency=currency,
        items=items,
        subtotal=subtotal,
        shipping=shipping,
        tax_label=tax_label,
        tax=tax,
        total=total,
        po_number=po_number,
        earlier_number=earlier_number,
        earlier_date=earlier_date,
        previous_balance=previous_balance,
        previous_date=previous_date,
        payment_terms=PAYMENT_TERMS[0] if level1 else draws.pick("terms", PAYMENT_TERMS),
        search_hits=_search_hits(draws, vendor_name, stem, trade, other_number),
        tax_name=tax_name,
        tax_rate=rate,
    )


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _format_date(date: datetime.date, style: str) -> str:
    if style == "iso":
        return date.isoformat()
    if style == "day_slash":
        return f"{date.day:02d}/{date.month:02d}/{date.year}"
    month = MONTH_NAMES[date.month - 1]
    if style == "day_month":
        return f"{date.day} {month} {date.year}"
    return f"{month} {date.day}, {date.year}"


def _format_number(value: Decimal, currency: str, difficulty: int) -> str:
    """Plain two-decimal form below level 3; grouped with the currency's marks at level 3."""
    if difficulty < 3:
        return f"{value:.2f}"
    grouped = f"{value:,.2f}"
    if currency == "EUR":
        return grouped.translate(_SWAP_DECIMAL_MARKS)
    return grouped


def _format_money(value: Decimal, currency: str, difficulty: int) -> str:
    """A totals-block amount: no symbol at level 1, the EUR sign after the number at level 3."""
    number = _format_number(value, currency, difficulty)
    if difficulty == 1:
        return number
    symbol = CURRENCY_SYMBOLS[currency]
    if difficulty >= 3 and currency == "EUR":
        return f"{number} {symbol}"
    return f"{symbol}{number}"


def _labelled_block(label: str, name: str, address: str) -> list[str]:
    """A name after its label, then the address aligned under the name."""
    return [f"{label}: {name}", " " * (len(label) + 2) + address]


def _header_lines(inv: _Invoice, block: str) -> list[str]:
    labels = inv.labels
    if block == "vendor":
        return _labelled_block(labels["vendor"], inv.vendor_name, inv.vendor_address)
    if block == "bill_to":
        return _labelled_block(labels["bill_to"], inv.customer_name, inv.customer_address)
    if block == "number":
        return [f"{labels['number']}: {inv.number}"]
    if block == "date":
        return [f"{labels['date']}: {_format_date(inv.date, inv.date_style)}"]
    if block == "currency":
        return [f"{labels['currency']}: {inv.currency}"]
    if block == "po_number":
        return [f"{labels['po_number']}: {inv.po_number}"]
    if block == "earlier_invoice" and inv.earlier_date is not None:
        earlier = _format_date(inv.earlier_date, inv.date_style)
        return [
            f"Note: our earlier invoice {inv.earlier_number} dated {earlier} "
            "was paid in full, thank you."
        ]
    raise ValueError(f"unknown header block {block!r}")


def _description_lines(inv: _Invoice, item: _LineItem) -> list[str]:
    """The description, wrapped at word boundaries at level 3 when longer than WRAP_WIDTH."""
    if inv.difficulty < 3 or len(item.description) <= WRAP_WIDTH:
        return [item.description]
    return textwrap.wrap(
        item.description, width=WRAP_WIDTH, break_long_words=False, break_on_hyphens=False
    )


def _item_cells(inv: _Invoice, item: _LineItem) -> list[str]:
    """The cells after the description: Qty, Unit price, [Disc.,] [Amount]. Level 4 prints
    no amount: the reader computes it."""
    cells = [str(item.quantity), _format_number(item.unit_price, inv.currency, inv.difficulty)]
    if inv.difficulty >= 3:
        cells.append(f"{item.discount}%" if item.discount else "")
    if inv.difficulty < 4:
        cells.append(_format_number(item.amount, inv.currency, inv.difficulty))
    return cells


def _table_lines(inv: _Invoice) -> list[str]:
    """Header, rule, one row per item (plus continuation lines), rule."""
    headers = [
        "Qty",
        "Unit price",
        *(["Disc."] if inv.difficulty >= 3 else []),
        *(["Amount"] if inv.difficulty < 4 else []),
    ]
    descriptions = [_description_lines(inv, item) for item in inv.items]
    cells = [_item_cells(inv, item) for item in inv.items]
    desc_width = max([len("Description")] + [len(line) for lines in descriptions for line in lines])
    widths = [max(len(row[k]) for row in [headers, *cells]) for k in range(len(headers))]

    def join(description: str, row: list[str]) -> str:
        values = [cell.rjust(width) for cell, width in zip(row, widths, strict=True)]
        return "  ".join([description.ljust(desc_width), *values])

    header = join("Description", headers)
    rule = "-" * len(header)
    lines = [header, rule]
    for description, row in zip(descriptions, cells, strict=True):
        lines.append(join(description[0], row))
        lines.extend(f"  {part}" for part in description[1:])
    lines.append(rule)
    return lines


def _totals_lines(inv: _Invoice, width: int) -> list[str]:
    """Label and amount lines, each column aligned, right-aligned to the table width. Level
    4 prints only the shipping charge and states how every other amount is computed."""
    if inv.difficulty >= 4:
        shipping = inv.labels["shipping"]
        charge = _format_money(inv.shipping or Decimal("0"), inv.currency, inv.difficulty)
        return [
            f"{shipping}: {charge}",
            "Each line amount is the quantity times the unit price, less the line's discount,",
            "rounded half up to the cent.",
            f"{inv.tax_name} at {inv.tax_rate}% is charged on the sum of the line amounts, rounded",
            f"half up to the cent; {shipping.lower()} is not taxed.",
            f"{inv.labels['total']}: the sum of the line amounts, plus {shipping.lower()}, plus "
            f"{inv.tax_name}.",
        ]
    rows: list[tuple[str, Decimal]] = []
    if inv.difficulty >= 2:
        rows.append(("Subtotal", inv.subtotal))
    if inv.shipping is not None:
        rows.append((inv.labels["shipping"], inv.shipping))
    if inv.tax is not None and inv.tax_label is not None:
        rows.append((inv.tax_label, inv.tax))
    rows.append((inv.labels["total"], inv.total))
    texts = [
        (f"{label}:", _format_money(value, inv.currency, inv.difficulty)) for label, value in rows
    ]
    label_width = max(len(label) for label, _ in texts)
    value_width = max(len(value) for _, value in texts)
    return [
        f"{label.rjust(label_width)} {value.rjust(value_width)}".rjust(width)
        for label, value in texts
    ]


def _footer_lines(inv: _Invoice) -> list[str]:
    lines: list[str] = []
    if inv.previous_balance is not None and inv.previous_date is not None:
        paid = _format_money(inv.previous_balance, inv.currency, inv.difficulty)
        received = _format_date(inv.previous_date, inv.date_style)
        lines.append(f"Previous balance: {paid} received on {received}, thank you.")
    lines.append(f"Payment terms: {inv.payment_terms}.")
    lines.append("Thank you for your business.")
    return lines


def _render(inv: _Invoice) -> str:
    """The document text: title, header, table, totals, footer, LF line endings."""
    lines = [inv.title, ""]
    for block in inv.header_order:
        lines.extend(_header_lines(inv, block))
    lines.append("")
    table = _table_lines(inv)
    lines.extend(table)
    lines.extend(_totals_lines(inv, len(table[0])))
    lines.append("")
    lines.extend(_footer_lines(inv))
    return "\n".join(lines) + "\n"


def _gold_record(inv: _Invoice) -> dict[str, Any]:
    """The gold record in the submit_record schema: ISO date, code, numbers as JSON numbers."""
    return {
        "invoice_number": inv.number,
        "invoice_date": inv.date.isoformat(),
        "vendor_name": inv.vendor_name,
        "currency": inv.currency,
        "total_amount": float(inv.total),
        "line_items": [
            {
                "description": item.description,
                "quantity": item.quantity,
                "unit_price": float(item.unit_price),
                "amount": float(item.amount),
            }
            for item in inv.items
        ],
    }


def _spreadsheet_rows(inv: _Invoice) -> list[dict[str, str]]:
    """The line items as generated, before any layout noise, as plain strings."""
    return [
        {
            "description": item.description,
            "quantity": str(item.quantity),
            "unit_price": f"{item.unit_price:.2f}",
            "amount": f"{item.amount:.2f}",
        }
        for item in inv.items
    ]


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def generate(seed: int, n: int, difficulty: int, out_dir: Path) -> None:
    """Write ``n`` documents, gold records, the search index and the lock to ``out_dir``.

    Refuses to overwrite a directory that already holds a LOCK.json (delete it deliberately).
    Also refuses, before writing anything, when ``out_dir`` holds a file this call would not
    write, so a lock never covers a stale file. Every generated value is drawn through
    rng.unit with the scopes listed in the module docstring; vendor names, descriptions and
    cities come from fixed word lists in this file.
    """
    out = Path(out_dir)
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"difficulty must be one of {DIFFICULTY_LEVELS}, got {difficulty}")
    if not 1 <= n <= MAX_DOCUMENTS:
        raise ValueError(f"n must be between 1 and {MAX_DOCUMENTS}, got {n}")
    lock_path = out / DATASET_LOCK_NAME
    if lock_path.exists():
        raise FileExistsError(f"{lock_path} exists; delete it deliberately to regenerate")
    doc_ids = [f"inv-{i:04d}" for i in range(1, n + 1)]
    expected = {f"{DOCUMENTS_DIR}/{doc_id}.txt" for doc_id in doc_ids}
    expected |= {GOLD_NAME, SEARCH_INDEX_NAME, SPREADSHEETS_NAME}
    present = {p.relative_to(out).as_posix() for p in _hashable_files(out)}
    stale = sorted(present - expected)
    if stale:
        raise FileExistsError(f"{out} holds files this generation would not write: {stale[:5]}")

    invoices = [_draw_invoice(seed, difficulty, i) for i in range(1, n + 1)]
    docs_dir = out / DOCUMENTS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    for inv in invoices:
        _write_text(docs_dir / f"{inv.doc_id}.txt", _render(inv))
    gold_lines = [
        json.dumps({"doc_id": inv.doc_id, "record": _gold_record(inv)}, sort_keys=True)
        for inv in invoices
    ]
    _write_text(out / GOLD_NAME, "\n".join(gold_lines) + "\n")
    _write_json(out / SEARCH_INDEX_NAME, {inv.doc_id: inv.search_hits for inv in invoices})
    _write_json(out / SPREADSHEETS_NAME, {inv.doc_id: _spreadsheet_rows(inv) for inv in invoices})
    write_lock(out, seed=seed, n=n, difficulty=difficulty)


def load(dataset_dir: Path) -> list[Document]:
    """Load every document, sorted by id, as UTF-8 with newlines normalized to LF."""
    docs_dir = Path(dataset_dir) / "documents"
    documents: list[Document] = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        documents.append(Document(id=path.stem, text=text))
    if not documents:
        raise FileNotFoundError(f"no documents under {docs_dir}")
    return documents


def load_gold(dataset_dir: Path) -> dict[str, GoldRecord]:
    """Load gold.jsonl keyed by doc_id."""
    path = Path(dataset_dir) / "gold.jsonl"
    gold: dict[str, GoldRecord] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gold[row["doc_id"]] = GoldRecord(doc_id=row["doc_id"], fields=row["record"])
    return gold


def _hashable_files(dataset_dir: Path) -> list[Path]:
    base = Path(dataset_dir)
    files = sorted(p for p in base.rglob("*") if p.is_file() and p.name != DATASET_LOCK_NAME)
    return files


def write_lock(dataset_dir: Path, *, seed: int, n: int, difficulty: int) -> dict[str, Any]:
    """Hash every file under the dataset (LF-normalized) and write LOCK.json."""
    base = Path(dataset_dir)
    files = {p.relative_to(base).as_posix(): sha256_file(p) for p in _hashable_files(base)}
    lock = {"schema_version": 1, "seed": seed, "n": n, "difficulty": difficulty, "files": files}
    (base / DATASET_LOCK_NAME).write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return lock


def verify_lock(dataset_dir: Path) -> dict[str, Any]:
    """Recompute every hash and raise LockMismatch naming the first file that differs,
    is missing, or is present but unlisted."""
    base = Path(dataset_dir)
    lock_path = base / DATASET_LOCK_NAME
    if not lock_path.exists():
        raise LockMismatch(f"{lock_path} is missing; run `islands dataset generate` first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected: dict[str, str] = lock["files"]
    actual = {p.relative_to(base).as_posix(): sha256_file(p) for p in _hashable_files(base)}
    for rel, digest in expected.items():
        if rel not in actual:
            raise LockMismatch(f"dataset file missing: {rel}")
        if actual[rel] != digest:
            raise LockMismatch(f"dataset file changed: {rel}")
    extra = sorted(set(actual) - set(expected))
    if extra:
        raise LockMismatch(f"dataset has unlisted files: {extra}")
    return lock
