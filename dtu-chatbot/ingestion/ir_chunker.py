#!/usr/bin/env python3
"""
Unified IR chunker — turns a Document IR (from any parser) into embed-ready
chunk records. Format-agnostic: it consumes blocks, not PDFs or HTML.

Design points carried over from the design discussion:
  * Contextual enrichment for free — every chunk is prefixed with its section
    breadcrumb ("DTU > Academics > Attendance"), so a retrieved fragment knows
    where it sits without an LLM call.
  * Tables are linearized row-by-row (Table.linearize) so each fee / grade /
    credit row survives retrieval as a self-contained fact, grouped only up to
    the token budget.
  * Prose is packed at sentence boundaries with overlap (reusing the proven
    primitives from chunker.py) so a rule spanning two chunks stays answerable.
  * Globally unique chunk IDs — sha256(source_url + chunk_index) — so chunks
    from different documents never collide in the vector DB.

The token-splitting primitives are imported from chunker.py rather than
duplicated. MAX_TOKENS here defaults to 512 (the tuning target from the
benchmarks); chunker.py keeps its historical 600 for the PDF path until that
path is migrated onto this chunker.

Usage (HTML end-to-end):
    python ingestion/ir_chunker.py data/raw/html/abc123.html \
        --url "https://dtu.ac.in/Web/Academics/scholarship.php" \
        --doc-type scholarship
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

# Ensure project root (dtu-chatbot/) is importable when run as a script
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import re as _re

from ingestion.chunker import _build_overlap, _split_sentences, count_tokens
from ingestion.document_ir import (
    Document,
    ListBlock,
    Paragraph,
    Section,
    Table,
)
from ingestion.html_parser import parse_html_file

MAX_TOKENS = 512          # tuning target; chunker.py PDF path still uses 600
MIN_TOKENS = 100          # merge trailing prose fragments below this
MIN_CHUNK_TOKENS = 15     # drop prose chunks below this (orphan section labels,
                          # e.g. "Ordinance / Regulations", "Former Principal")

_YEAR_RE = _re.compile(r"20(1[5-9]|2[0-9])")   # 2015–2029


def _batch_year(doc: Document) -> int:
    """
    Extract a 4-digit batch year from document metadata.

    Checks date_published → title → url in order. Returns 0 when no year
    is found (HTML dept pages, evergreen content, etc.).
    """
    for text in (doc.date_published or "", doc.title or "", doc.url or ""):
        m = _YEAR_RE.search(text)
        if m:
            return int(m.group(0))
    return 0


# ---------------------------------------------------------------------------
# Prose packing (sentence-aware, with overlap)
# ---------------------------------------------------------------------------


def _pack_prose(text: str, max_tokens: int) -> list[str]:
    """Split prose into <=max_tokens pieces at sentence boundaries, with overlap."""
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    # First pass: greedy grouping at the token budget.
    groups: list[list[str]] = []
    current: list[str] = []
    current_tok = 0
    for s in sentences:
        s_tok = count_tokens(s)
        if current and current_tok + s_tok + 1 > max_tokens:
            groups.append(current)
            current = [s]
            current_tok = s_tok
        else:
            current.append(s)
            current_tok += s_tok + 1
    if current:
        groups.append(current)

    # Second pass: prepend overlap from the previous group.
    pieces: list[str] = []
    for i, group in enumerate(groups):
        piece = " ".join(group)
        if i > 0:
            overlap = _build_overlap(" ".join(groups[i - 1]))
            if overlap:
                piece = overlap + " " + piece
        pieces.append(piece)

    # Merge a tiny trailing piece back into its predecessor.
    merged: list[str] = []
    for piece in pieces:
        if merged and count_tokens(piece) < MIN_TOKENS:
            merged[-1] = merged[-1] + " " + piece
        else:
            merged.append(piece)
    return merged


def _group_table_rows(table: Table, max_tokens: int, breadcrumb: str) -> list[str]:
    """Group linearized rows into blobs that fit the budget (with breadcrumb)."""
    rows = table.linearize()
    base = count_tokens(breadcrumb) + 2
    groups: list[str] = []
    current: list[str] = []
    current_tok = base
    for r in rows:
        r_tok = count_tokens(r) + 1
        if current and current_tok + r_tok > max_tokens:
            groups.append("\n".join(current))
            current = [r]
            current_tok = base + r_tok
        else:
            current.append(r)
            current_tok += r_tok
    if current:
        groups.append("\n".join(current))
    return groups


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_id(source_url: str, chunk_index: int) -> str:
    """Globally unique, deterministic id — survives across documents and reruns."""
    digest = hashlib.sha256(f"{source_url}::{chunk_index}".encode()).hexdigest()
    return f"c_{digest[:14]}"


def _record(
    doc: Document,
    section: Section,
    breadcrumb: str,
    content: str,
    block_type: str,
    chunk_index: int,
    date_scraped: str,
) -> dict:
    text = f"{breadcrumb}\n\n{content}"
    return {
        "chunk_id": _chunk_id(doc.url, chunk_index),
        "text": text,
        "source_url": doc.url,
        "document_title": doc.title,
        "section_heading": section.heading,
        "breadcrumb": breadcrumb,
        "document_type": doc.doc_type,
        # Empty string, not None: vector-DB metadata (ChromaDB/Pinecone) rejects
        # null values. "" reads as "date unknown".
        "date_published": doc.date_published or "",
        "date_scraped": date_scraped,
        "chunk_index": chunk_index,
        "token_count": count_tokens(text),
        "source_format": doc.source_format,
        "block_type": block_type,
        "page_number": section.page,          # int for PDF, None for HTML
        "batch_year": _batch_year(doc),       # 0 = not year-specific (HTML, evergreen docs)
    }


def chunk_document(
    doc: Document,
    *,
    date_scraped: str | None = None,
    max_tokens: int = MAX_TOKENS,
) -> list[dict]:
    """Produce chunk records for a whole Document IR."""
    date_scraped = date_scraped or date.today().isoformat()
    records: list[dict] = []
    idx = 0

    for section in doc.iter_sections():
        breadcrumb = section.breadcrumb_str
        prose_buffer: list[str] = []

        def flush_prose() -> None:
            nonlocal idx
            if not prose_buffer:
                return
            blob = "\n".join(prose_buffer).strip()
            prose_buffer.clear()
            # The DTU CMS renders the page title as body text too, so it shows up
            # both in the breadcrumb and at the start of the prose. Strip the
            # leading repeat — this also drops title-only "chunks" entirely.
            head = section.heading.strip()
            if head and blob.startswith(head):
                blob = blob[len(head):].lstrip(" :-\n")
            if not blob:
                return
            bc_overhead = count_tokens(breadcrumb) + 2  # mirrors _group_table_rows
            for piece in _pack_prose(blob, max_tokens - bc_overhead):
                # Skip orphan-label fragments (a section whose only "prose" is its
                # own sub-heading). Tables are never floored — short tables like
                # the contact block are still meaningful.
                if count_tokens(piece) < MIN_CHUNK_TOKENS:
                    continue
                records.append(
                    _record(doc, section, breadcrumb, piece, "prose", idx, date_scraped)
                )
                idx += 1

        # Faculty pages: one person = one chunk for precise name retrieval.
        one_per_block = doc.doc_type == "dept_faculty"

        for block in section.blocks:
            if isinstance(block, Table):
                flush_prose()                         # keep document order
                for blob in _group_table_rows(block, max_tokens, breadcrumb):
                    records.append(
                        _record(doc, section, breadcrumb, blob, "table", idx, date_scraped)
                    )
                    idx += 1
            elif isinstance(block, ListBlock):
                prose_buffer.append(block.as_text())
            elif isinstance(block, Paragraph):
                prose_buffer.append(block.text)
                if one_per_block:
                    flush_prose()

        flush_prose()

    # Content dedup: spread-layout PDFs and CMS template repetition can emit
    # the same text under two chunk ids — keep the first occurrence only.
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in records:
        h = hashlib.sha256(r["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        deduped.append(r)

    return deduped


# ---------------------------------------------------------------------------
# Main — HTML end-to-end
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk a Document IR (HTML) into JSONL")
    ap.add_argument("html_path", type=Path, help="Saved HTML body to chunk")
    ap.add_argument("--url", required=True, help="Original source URL")
    ap.add_argument("--doc-type", dest="doc_type", default="unknown")
    ap.add_argument("--title", default=None)
    ap.add_argument("--date", dest="date_published", default=None)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    if not args.html_path.exists():
        print(f"File not found: {args.html_path}", file=sys.stderr)
        sys.exit(1)

    doc = parse_html_file(
        args.html_path,
        url=args.url,
        doc_type=args.doc_type,
        title=args.title,
        date_published=args.date_published,
    )
    chunks = chunk_document(doc, max_tokens=args.max_tokens)

    out_dir = Path("data/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.html_path.stem}_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["block_type"]] = by_type.get(c["block_type"], 0) + 1
    tokens = [c["token_count"] for c in chunks]

    print("================================")
    print("IR CHUNKING COMPLETE")
    print("================================")
    print(f"Source         : {doc.url}")
    print(f"Title          : {doc.title}")
    print(f"Format         : {doc.source_format}")
    print(f"Chunks created : {len(chunks)}  {by_type}")
    if tokens:
        print(f"Token sizes    : min={min(tokens)} avg={sum(tokens)//len(tokens)} max={max(tokens)}")
    print(f"Output         : {out_path}")


if __name__ == "__main__":
    main()
