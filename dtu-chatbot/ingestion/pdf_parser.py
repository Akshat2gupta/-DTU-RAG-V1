#!/usr/bin/env python3
"""
PDF → Document IR parser.

The parallel of html_parser.py for PDFs. Uses chunker.py's hard-won pipeline
for page extraction (classifier filter), font + 14-rule heading detection, and
section building — and converts the resulting sections into the shared IR blocks.

Tables: replaced whitespace-heuristic column detection with pdfplumber's
page.extract_tables(), which reads actual grid lines and returns structured
rows. Each table's first row becomes headers; merged cells (None) inherit from
the row above. The first section on each page claims that page's tables.

Usage:
    python ingestion/pdf_parser.py data/raw/pdfs/BTech_2022_ordinance.pdf \
        --url "https://dtu.ac.in/.../BTech_2022_ordinance.pdf" \
        --title "DTU B.Tech Ordinance 2022" --doc-type ordinance --date 2022-01-01
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Ensure project root (dtu-chatbot/) is importable when run as a script
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import pdfplumber

from ingestion.chunker import (
    MIN_SECTION_TOKENS,
    _POLICY_KEYWORDS_RE,
    build_sections,
    count_tokens,
    detect_headings,
    extract_pages,
)
from ingestion.document_ir import Block, Document, Heading, Paragraph, Table

_TOC_LINE_RE = re.compile(r"\b\d{1,3}\s*$")        # entry ending in a page number


def _is_toc_section(section: dict) -> bool:
    """True when most lines end in a bare page number (table-of-contents page)."""
    lines = section["content_lines"]
    if len(lines) < 4:
        return False
    ending_in_num = sum(1 for ln in lines if _TOC_LINE_RE.search(ln))
    return ending_in_num >= 0.6 * len(lines)


def _keep_section(section: dict) -> bool:
    """Port of chunker.py's thin-section gate, plus TOC rejection."""
    if _is_toc_section(section):
        return False
    if count_tokens(section["raw_text"]) >= MIN_SECTION_TOKENS:
        return True
    return bool(_POLICY_KEYWORDS_RE.search(section["heading"]))


# ---------------------------------------------------------------------------
# Real table extraction (pdfplumber grid detector)
# ---------------------------------------------------------------------------


def _fill_merged_cells(raw_rows: list[list]) -> list[list[str]]:
    """
    pdfplumber returns None for cells that are part of a merged region.
    Fill each such cell by inheriting the value from the same column in
    the row above, so every data row is self-contained.
    """
    if not raw_rows:
        return []
    col_count = max(len(r) for r in raw_rows)
    prev = [""] * col_count
    result: list[list[str]] = []
    for raw_row in raw_rows:
        row: list[str] = []
        for j in range(col_count):
            cell = raw_row[j] if j < len(raw_row) else None
            val = (cell or "").strip()
            if not val:
                val = prev[j]        # inherit from above
            row.append(val)
        prev = row[:]
        result.append(row)
    return result


def _raw_table_to_ir(raw_table: list[list], page: int) -> Table | None:
    """
    Convert a pdfplumber raw table (list-of-lists) to an IR Table block.

    Row 0  → headers (blank cells become "Col N").
    Rows 1+ → data rows; fully-empty rows are skipped.
    Returns None when the table has fewer than 2 rows (header only).
    """
    if not raw_table or len(raw_table) < 2:
        return None

    filled = _fill_merged_cells(raw_table)

    headers: list[str] = []
    for j, h in enumerate(filled[0]):
        h = (h or "").strip()
        headers.append(h if h else f"Col {j + 1}")

    data_rows: list[list[str]] = []
    for row in filled[1:]:
        # Pad short rows, truncate long rows to header width
        padded = (row + [""] * len(headers))[: len(headers)]
        cells = [c.strip() for c in padded]
        if any(cells):          # skip fully-empty rows
            data_rows.append(cells)

    if not data_rows:
        return None

    return Table(rows=data_rows, headers=headers, page=page)


def _extract_page_tables(
    pdf_path: Path,
    allowed_pages: set[int],
) -> dict[int, list[Table]]:
    """
    Open the PDF with pdfplumber and extract real tables from every page in
    *allowed_pages* (pages that passed extract_pages()'s classifier filter).

    Returns {page_number: [Table, ...]} — only pages that have ≥1 real table.
    """
    result: dict[int, list[Table]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for n, page in enumerate(pdf.pages, start=1):
            if n not in allowed_pages:
                continue
            raw_tables = page.extract_tables() or []
            ir_tables = [
                t
                for raw in raw_tables
                if (t := _raw_table_to_ir(raw, page=n)) is not None
            ]
            if ir_tables:
                result[n] = ir_tables
    return result


# ---------------------------------------------------------------------------
# Section → IR blocks
# ---------------------------------------------------------------------------


def _section_to_blocks(
    section: dict,
    page_tables: dict[int, list[Table]],
    tables_emitted: set[int],
) -> list[Block]:
    """
    Convert one chunker.py section dict into IR blocks.

    Prose lines are joined into a single Paragraph.  Real tables (from
    pdfplumber's page.extract_tables()) are appended after the prose; the
    *first* section whose page_number matches a page in page_tables claims
    that page's tables (tracked via tables_emitted to avoid double-emit).
    """
    heading = section["heading"]
    page    = section["page_number"]
    lines   = section["content_lines"]
    blocks: list[Block] = []

    # "Preamble" is chunker.py's placeholder for content before the first
    # real heading; emit as leading prose with no Heading block.
    if heading and heading != "Preamble":
        blocks.append(Heading(text=heading, level=1, page=page))

    prose = " ".join(lines).strip()
    if prose:
        blocks.append(Paragraph(text=prose, page=page))

    # Claim this page's tables (once per page — first section on the page wins)
    if page not in tables_emitted and page in page_tables:
        blocks.extend(page_tables[page])
        tables_emitted.add(page)

    return blocks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pdf(
    pdf_path: Path,
    url: str,
    doc_type: str = "ordinance",
    title: str | None = None,
    date_published: str | None = None,
) -> Document:
    """Parse a PDF into a Document IR using chunker.py's extraction pipeline
    for headings/sections and pdfplumber's grid detector for tables."""
    pdf_path = Path(pdf_path)
    pages, _skipped, _total, _class_counts = extract_pages(pdf_path)
    headings  = detect_headings(pages)
    sections  = build_sections(pages, headings)
    sections  = [s for s in sections if _keep_section(s)]

    # Second pdfplumber pass — structured table extraction on classifier-approved pages
    allowed     = {p["page_number"] for p in pages}
    page_tables = _extract_page_tables(pdf_path, allowed)

    blocks: list[Block] = []
    tables_emitted: set[int] = set()
    for section in sections:
        blocks.extend(_section_to_blocks(section, page_tables, tables_emitted))

    return Document(
        url=url,
        title=title or pdf_path.stem,
        source_format="pdf",
        doc_type=doc_type,
        date_published=date_published,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF -> Document IR -> chunks")
    ap.add_argument("pdf_path", type=Path)
    ap.add_argument("--url", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--doc-type", dest="doc_type", default="ordinance")
    ap.add_argument("--date", dest="date_published", default=None)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    args = ap.parse_args()

    if not args.pdf_path.exists():
        print(f"File not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {args.pdf_path.name} ...")
    doc = parse_pdf(
        args.pdf_path,
        url=args.url,
        doc_type=args.doc_type,
        title=args.title,
        date_published=args.date_published,
    )

    from ingestion.ir_chunker import chunk_document

    kwargs = {"max_tokens": args.max_tokens} if args.max_tokens else {}
    chunks = chunk_document(doc, **kwargs)

    out_dir = Path("data/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.pdf_path.stem}_ir_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Summary
    kinds: dict[str, int] = {}
    for b in doc.blocks:
        kinds[b.kind] = kinds.get(b.kind, 0) + 1

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["block_type"]] = by_type.get(c["block_type"], 0) + 1

    toks = [c["token_count"] for c in chunks]
    pages_seen = {c["page_number"] for c in chunks if c["page_number"]}

    table_pages = sorted(kinds_page for kinds_page in {
        b.page for b in doc.blocks if b.kind == "table"
    } if kinds_page)

    print("================================")
    print("PDF -> IR CHUNKING COMPLETE")
    print("================================")
    print(f"Title        : {doc.title}")
    print(f"IR blocks    : {len(doc.blocks)}  {kinds}")
    print(f"Chunks       : {len(chunks)}  {by_type}")
    if toks:
        print(f"Token sizes  : min={min(toks)} avg={sum(toks)//len(toks)} max={max(toks)}")
    print(f"Pages covered: {len(pages_seen)}")
    print(f"Table blocks : {kinds.get('table', 0)} tables on pages {table_pages}")
    print(f"Output       : {out_path}")


if __name__ == "__main__":
    main()
