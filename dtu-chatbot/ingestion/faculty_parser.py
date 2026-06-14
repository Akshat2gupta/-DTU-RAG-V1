#!/usr/bin/env python3
"""
Faculty page parser — extracts structured faculty records from DTU's CMS
table layout and returns a Document IR.

DTU faculty pages use a nested-table layout where each faculty member
occupies one <td> cell in the form:

    Name  Designation: X  Qualification: Y  Specialization: Z  Email: …

Section-header rows ("Professor", "Associate Professor", etc.) separate
rank groups.  This parser turns each faculty record into a self-contained
Paragraph so RAG retrieval can answer questions like "who teaches AI at
CSE?" or "list professors in Electronics".

Usage (standalone):
    cd dtu-chatbot/
    python ingestion/faculty_parser.py data/raw/html/<hash>.html \
        --url "https://dtu.ac.in/Web/Departments/CSE/faculty/" \
        --dept CSE
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import lxml.html

from ingestion.document_ir import Document, Heading, Paragraph

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_SECTION_HEADERS = frozenset({
    "head of department", "associate head of department",
    "professor", "associate professor", "assistant professor",
    "programmer", "technical staff", "administrative staff",
    "lecturer", "senior lecturer",
})

_EMAIL_RE = re.compile(r"\[at\]", re.IGNORECASE)
_DOT_RE   = re.compile(r"\[dot\]", re.IGNORECASE)

# A cell is a faculty record if it contains "Designation:"
_FACULTY_MARKER = re.compile(r"Designation\s*:", re.IGNORECASE)

# Extract named fields from the flat cell text
_FIELD_RE = re.compile(
    r"(?:Designation|Qualification|Specialization|Email|Research\s*Interest)"
    r"\s*:\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = " ".join(text.split())
    text = _EMAIL_RE.sub("@", text)
    text = _DOT_RE.sub(".", text)
    return text.strip()


def _is_section_header(text: str) -> bool:
    return text.strip().lower() in _SECTION_HEADERS


def _parse_faculty_cell(raw: str) -> str:
    """
    Turn the flat cell text into a readable sentence:
    "Prof. X — Professor — Specialization: Y — Email: Z"
    """
    raw = _clean(raw)

    # Split at field labels to separate the name from the rest
    parts = _FIELD_RE.split(raw)
    name  = parts[0].strip().rstrip("-–—").strip()

    # Re-join labelled fields
    labels = _FIELD_RE.findall(raw)
    fields: list[str] = []
    for label, value in zip(labels, parts[1:]):
        label = label.strip().rstrip(":").strip()
        value = value.strip()
        if value:
            fields.append(f"{label}: {value}")

    if fields:
        return f"{name} — {' — '.join(fields)}"
    return name


def _find_faculty_table(tree) -> lxml.html.HtmlElement | None:
    """
    Return the innermost <table> that contains faculty records.

    DTU pages use nested layout tables: page nav → content → faculty rows.
    A parent table scores higher on "Designation:" count than its child because
    cssselect("td") traverses descendants. We want the innermost candidate —
    the one that has no child table that is also a candidate.
    """
    all_tables    = tree.cssselect("table")
    candidates    = [
        tbl for tbl in all_tables
        if any(_FACULTY_MARKER.search(td.text_content()) for td in tbl.cssselect("td"))
    ]
    if not candidates:
        return None

    candidate_ids = {id(t) for t in candidates}
    for tbl in candidates:
        # Use XPath .//table (not cssselect) — cssselect("table") on a <table>
        # element returns the element itself in lxml, making every table appear
        # to have itself as a child candidate.
        child_candidate = any(
            id(ct) in candidate_ids for ct in tbl.xpath(".//table")
        )
        if not child_candidate:
            return tbl   # innermost: no descendant table is also a candidate

    return candidates[0]  # fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_faculty_html(
    path: Path,
    url: str,
    dept: str | None = None,
    doc_type: str = "dept_faculty",
) -> Document:
    """
    Parse a DTU faculty HTML page into a Document IR.

    Each faculty member becomes one Paragraph.  Section headers
    ("Professor", "Associate Professor", …) become Headings.
    """
    html  = Path(path).read_text(encoding="utf-8", errors="replace")
    tree  = lxml.html.fromstring(html)
    title = (dept or "Faculty") + " — Faculty"

    tbl = _find_faculty_table(tree)

    blocks = [Heading(text=title, level=1)]

    if tbl is None:
        # Fallback: no recognisable faculty table — return empty doc
        return Document(url=url, title=title, source_format="html",
                        doc_type=doc_type, blocks=blocks)

    current_section: str | None = None

    for row in tbl.cssselect("tr"):
        cells = [td for td in row.cssselect("td, th")]
        if not cells:
            continue

        # Flatten all cell text into one string (some rows span multiple cells)
        cell_texts = [_clean(c.text_content()) for c in cells]
        combined   = " ".join(t for t in cell_texts if t)

        if not combined:
            continue

        # --- Section header row ---
        if _is_section_header(combined):
            current_section = combined.title()
            blocks.append(Heading(text=current_section, level=2))
            continue

        # --- Faculty record (contains "Designation:") ---
        if _FACULTY_MARKER.search(combined):
            record = _parse_faculty_cell(combined)
            if dept:
                record += f" — Department: {dept}"
            blocks.append(Paragraph(text=record))
            continue

        # --- Skip nav / boilerplate rows (very long or no alpha content) ---

    # DTU's CMS sometimes emits <td> directly under <table> without a <tr>
    # wrapper (malformed HTML). These entries are invisible to the TR loop above.
    for td in tbl:
        if td.tag != "td":
            continue
        combined = _clean(td.text_content())
        if not combined or not _FACULTY_MARKER.search(combined):
            continue
        record = _parse_faculty_cell(combined)
        if dept:
            record += f" — Department: {dept}"
        blocks.append(Paragraph(text=record))

    return Document(
        url=url,
        title=title,
        source_format="html",
        doc_type=doc_type,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Parse DTU faculty HTML page")
    ap.add_argument("html_path", type=Path)
    ap.add_argument("--url",   required=True)
    ap.add_argument("--dept",  default=None, help="Department name (e.g. CSE)")
    args = ap.parse_args()

    doc = parse_faculty_html(args.html_path, url=args.url, dept=args.dept)

    print(f"Title  : {doc.title}")
    print(f"Blocks : {len(doc.blocks)}")
    for b in doc.blocks:
        txt = getattr(b, "text", "")
        print(f"  [{b.kind:9s}] {txt[:120]}")


if __name__ == "__main__":
    main()
