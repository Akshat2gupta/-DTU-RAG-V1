#!/usr/bin/env python3
"""
DTU Chunk Validator — validates JSONL output from chunker.py.

Usage:
    python ingestion/test_chunker.py data/chunks/BTech_2022_ordinance_chunks.jsonl

Runs all 10 checks, reports ALL failures (never stops at first).
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

REQUIRED_FIELDS = {
    "chunk_id", "text", "source_url", "document_title",
    "section_heading", "document_type", "date_published",
    "date_scraped", "chunk_index", "token_count", "is_ocr", "page_number",
}

_PURELY_NUMERIC_RE = re.compile(r"^\d+$")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python ingestion/test_chunker.py <path_to_jsonl>")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    issues: list[str] = []

    # ── CHECK 1: File exists and is valid JSONL ───────────────────────────────
    print("─" * 60)
    if not jsonl_path.exists():
        print(f"✗ FAILED — File not found: {jsonl_path}")
        sys.exit(1)

    chunks: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                chunks.append(json.loads(raw))
            except json.JSONDecodeError as e:
                issues.append(f"Line {line_num}: invalid JSON — {e}")

    print(f"CHECK 1  — Valid JSONL       : {len(chunks)} chunks loaded")

    if not chunks:
        print("✗ FAILED — No chunks found in file.")
        sys.exit(1)

    # ── CHECK 2: Required fields ──────────────────────────────────────────────
    missing_field_count = 0
    for chunk in chunks:
        cid     = chunk.get("chunk_id", f"<unknown>")
        missing = REQUIRED_FIELDS - set(chunk.keys())
        if missing:
            issues.append(f"{cid}: missing fields — {', '.join(sorted(missing))}")
            missing_field_count += 1
    print(f"CHECK 2  — Required fields   : {missing_field_count} chunks with missing fields")

    # ── CHECK 3: Token count accuracy (±5 tolerance) ──────────────────────────
    tok_mismatches = 0
    for chunk in chunks:
        cid    = chunk.get("chunk_id", "?")
        stored = chunk.get("token_count", 0)
        actual = len(enc.encode(chunk.get("text", "")))
        if abs(stored - actual) > 5:
            issues.append(
                f"{cid}: token_count mismatch — stored {stored}, recomputed {actual}"
            )
            tok_mismatches += 1
    print(f"CHECK 3  — Token accuracy    : {tok_mismatches} mismatches (tolerance ±5)")

    # ── CHECK 4: Size bounds 50–700 tokens ────────────────────────────────────
    size_violations = 0
    for chunk in chunks:
        cid = chunk.get("chunk_id", "?")
        tok = len(enc.encode(chunk.get("text", "")))
        if tok < 50:
            issues.append(f"{cid}: under 50 tokens ({tok})")
            size_violations += 1
        elif tok > 700:
            issues.append(f"{cid}: over 700 tokens ({tok})")
            size_violations += 1
    print(f"CHECK 4  — Size bounds 50–700: {size_violations} violations")

    # ── CHECK 5: Section heading quality ──────────────────────────────────────
    heading_violations = 0
    for chunk in chunks:
        cid = chunk.get("chunk_id", "?")
        sh  = chunk.get("section_heading", None)
        if sh is None or not str(sh).strip():
            issues.append(f"{cid}: section_heading is empty or whitespace")
            heading_violations += 1
        elif _PURELY_NUMERIC_RE.match(str(sh).strip()):
            issues.append(f"{cid}: section_heading is purely numeric: {sh!r}")
            heading_violations += 1
        elif len(str(sh).strip()) < 4:
            issues.append(f"{cid}: section_heading under 4 chars: {sh!r}")
            heading_violations += 1
    print(f"CHECK 5  — Heading quality   : {heading_violations} violations")

    # ── CHECK 6: Text field starts with "Section: " ───────────────────────────
    prefix_violations = 0
    for chunk in chunks:
        cid = chunk.get("chunk_id", "?")
        if not chunk.get("text", "").startswith("Section: "):
            issues.append(f"{cid}: text does not start with 'Section: '")
            prefix_violations += 1
    print(f"CHECK 6  — Section prefix    : {prefix_violations} chunks missing it")

    # ── CHECK 7: chunk_id sequential, no gaps, no duplicates ──────────────────
    seen_ids = [c.get("chunk_id", "") for c in chunks]

    id_counts: dict[str, int] = defaultdict(int)
    for cid in seen_ids:
        id_counts[cid] += 1
    dup_ids = [cid for cid, cnt in id_counts.items() if cnt > 1]
    for cid in dup_ids:
        issues.append(f"Duplicate chunk_id: {cid} (appears {id_counts[cid]} times)")

    # Check sequential from auto_001
    first_gap: str | None = None
    for i, cid in enumerate(seen_ids, start=1):
        expected = f"auto_{i:03d}"
        if cid != expected:
            first_gap = f"Expected {expected}, got {cid!r}"
            issues.append(f"chunk_id gap/mismatch at position {i}: {first_gap}")
            break  # one report is enough — subsequent gaps are noise

    seq_issues = len(dup_ids) + (1 if first_gap else 0)
    print(f"CHECK 7  — chunk_id sequence : {seq_issues} issues")

    # ── CHECK 8: No duplicate text ────────────────────────────────────────────
    text_to_ids: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        text_to_ids[chunk.get("text", "")].append(chunk.get("chunk_id", "?"))
    dup_texts = {k: v for k, v in text_to_ids.items() if len(v) > 1}
    for text, cids in dup_texts.items():
        issues.append(f"Duplicate text in chunks: {', '.join(cids)}")
    print(f"CHECK 8  — No duplicate text : {len(dup_texts)} duplicate text groups")

    # ── CHECK 9: chunk_index consistency within each section_heading ──────────
    # Group by (source_url, section_heading, page_number) to avoid merging
    # same-named headings that appear on different pages of the document.
    sections: dict[tuple, list[dict]] = defaultdict(list)
    for chunk in chunks:
        key = (
            chunk.get("source_url", ""),
            chunk.get("section_heading", ""),
            chunk.get("page_number", 0),
        )
        sections[key].append(chunk)

    idx_violations = 0
    for heading, sec_chunks in sections.items():
        ordered = sorted(sec_chunks, key=lambda c: c.get("chunk_id", ""))
        for expected_idx, chunk in enumerate(ordered):
            actual_idx = chunk.get("chunk_index", -1)
            if actual_idx != expected_idx:
                issues.append(
                    f"{chunk.get('chunk_id', '?')}: chunk_index={actual_idx}, "
                    f"expected {expected_idx} in section '{heading}'"
                )
                idx_violations += 1
    print(f"CHECK 9  — chunk_index order : {idx_violations} violations")

    # ── CHECK 10: Manual review sample ────────────────────────────────────────
    sample_n = min(5, len(chunks))
    sample   = random.sample(chunks, sample_n)
    print(f"\nCHECK 10 — Manual review sample ({sample_n} random chunks):")
    for chunk in sample:
        preview = (chunk.get("text") or "")[:200]
        print(f"\n{'─' * 65}")
        print(f"chunk_id    : {chunk.get('chunk_id', '?')}")
        print(f"section     : {chunk.get('section_heading', '?')}")
        print(f"page        : {chunk.get('page_number', '?')}")
        print(f"tokens      : {chunk.get('token_count', '?')}")
        print(f"chunk_index : {chunk.get('chunk_index', '?')}")
        print(f"text preview: {preview}...")
    print("─" * 65)

    # ── Final result ──────────────────────────────────────────────────────────
    print()
    if not issues:
        print(f"✓ PASSED — {len(chunks)} chunks validated successfully")
        sys.exit(0)
    else:
        print(f"✗ FAILED — {len(issues)} issues found:")
        for issue in issues:
            print(f"  • {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
