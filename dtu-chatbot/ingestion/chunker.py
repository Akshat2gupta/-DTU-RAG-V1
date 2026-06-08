#!/usr/bin/env python3
"""
DTU PDF Chunker — section-aware semantic chunking.

Usage:
    python ingestion/chunker.py data/raw/pdfs/BTech_2022_ordinance.pdf \
        --url "https://dtu.ac.in/Web/Academics/ordinance/BTech_2022_ordinance.pdf" \
        --title "DTU B.Tech Academic Ordinance and Regulations 2022" \
        --doc-type "ordinance" \
        --date "2022-01-01"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# Ensure project root (dtu-chatbot/) is importable when run as a script
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from ingestion.classifier import classify_page

import pdfplumber
import tiktoken

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

enc = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS       = 600
MIN_TOKENS       = 100
TABLE_MAX_TOKENS = 400
OVERLAP_TARGET   = 100
MAX_OVERLAP_TOKENS  = 120
MIN_SECTION_TOKENS  = 150   # fix 4: discard thin sections below this threshold

_POLICY_KEYWORDS_RE = re.compile(
    r"\b(?:attendance|grade|grading|examination|hostel|fee|scholarship|"
    r"promotion|withdrawal|registration|credit|cgpa)\b",
    re.IGNORECASE,
)

_PURELY_NUMERIC_RE = re.compile(r"^\d+\.?\d*$")
_BARE_SUBSEC_RE    = re.compile(r"^\d+\.\d+\s*$")
_SUBLIST_RE        = re.compile(r"^(?:[a-d]\)|(?:i{1,3}|iv|vi{0,3}|ix|x{1,3})\))", re.IGNORECASE)
_COL_ALIGN_RE      = re.compile(r"\S+\s{3,}\S+")

# Bug 2: additional heading exclusion patterns
_TRAILING_PREP_RE  = re.compile(
    r"\b(?:and|or|the|a|an|of|in|for|with|to|from|by|as|at|on|into|"
    r"about|that|which|who|whose|when|where|if|but|nor|yet|so)\s*$",
    re.IGNORECASE,
)
_SENTENCE_VERB_RE  = re.compile(
    r"\b(?:shall|must|will|may|is|are|was|were)\b", re.IGNORECASE
)
_STARTS_NUM_ROMAN_RE = re.compile(r"^(?:\d|[IVXLCDM]+\.?\s)", re.IGNORECASE)
_REGULATION_HDR_RE   = re.compile(r"^R\.\s*1\s*\(B\)", re.IGNORECASE)
# Broader: matches any R.N(X).N regulation heading (R.1(A).3, R.1(B).22, R.2.1 …)
_REGULATION_NUM_RE   = re.compile(
    r"^R\.\s*\d+\s*(?:\([A-Za-z]\))?\s*\.\s*\d+", re.IGNORECASE
)
_DEFN_ENTRY_RE       = re.compile(
    r"^(?:[ivxlcdm]+\.|[a-z]\.)\s+.+\bshall mean\b", re.IGNORECASE
)
# Junk-heading filters (applied after Method A/B detection)
_MATH_SYMBOLS_RE = re.compile(r"[=∑σ≥≤±×÷→←∫∝∂]")
_DOT_FILLER_RE   = re.compile(r"[.…]{3,}|[-–—]{4,}")

# Bug 3: abbreviation-aware sentence splitting
_ABBREVS_RE = re.compile(
    r"\b(?:Ph|M|B|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|"
    r"No|Sec|Reg|Vol|vs|etc|approx|dept|univ|"
    r"govt|viz|i\.e|e\.g)\.",
    re.IGNORECASE,
)


def _is_junk_heading(line: str) -> bool:
    """True if *line* is a formula fragment or table-cell filler, not a real heading."""
    if _MATH_SYMBOLS_RE.search(line):
        return True
    if _DOT_FILLER_RE.search(line):
        return True
    return False


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


# =============================================================================
# STEP 1 — Extract pages
# =============================================================================

def extract_pages(pdf_path: Path) -> tuple[list[dict], list[int], int, dict[str, int]]:
    """Return (processed_pages, skipped_page_nums, total_page_count, class_counts)."""
    pages: list[dict] = []
    skipped: list[int] = []
    class_counts: dict[str, int] = {
        "policy": 0, "notice": 0, "syllabus": 0,
        "contact": 0, "form": 0, "skip": 0,
    }

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for n, page in enumerate(pdf.pages, start=1):
            # Bug 1: skip ghost duplicate pages where all content sits outside the cropbox
            words = page.extract_words()
            if words and all(w["x0"] > page.width for w in words):
                log.info(f"Page {n}: skipped (ghost duplicate — all content outside cropbox)")
                skipped.append(n)
                continue

            text = page.extract_text() or ""
            if len(text.strip()) < 100:
                log.info(f"Page {n}: skipped (likely scanned)")
                skipped.append(n)
                class_counts["skip"] += 1
                continue

            page_type = classify_page(text)
            class_counts[page_type] += 1

            if page_type in ("syllabus", "contact", "form", "skip"):
                log.info(f"Page {n}: skipped (classified as {page_type})")
                skipped.append(n)
                continue

            chars = page.chars or []
            sizes = [c["size"] for c in chars if c.get("size", 0) > 0]
            avg_font = sum(sizes) / len(sizes) if sizes else 0.0

            pages.append({
                "page_number": n,
                "raw_text":    text,
                "chars":       chars,
                "avg_font_size": avg_font,
            })

    return pages, skipped, total, class_counts


# =============================================================================
# STEP 2 — Detect section headings
# =============================================================================

def _build_line_font_map(chars: list[dict]) -> dict[str, float]:
    """
    Group chars by y-coordinate (2 pt bins) → average font size per line.
    Line text is the key; last-write wins on collision (rare for real headings).
    """
    buckets: dict[int, list[dict]] = defaultdict(list)
    for c in chars:
        y_key = round(c.get("top", 0) / 2) * 2
        buckets[y_key].append(c)

    result: dict[str, float] = {}
    for char_list in buckets.values():
        line_text = "".join(c.get("text", "") for c in char_list).strip()
        if not line_text:
            continue
        sizes = [c["size"] for c in char_list if c.get("size", 0) > 0]
        result[line_text] = sum(sizes) / len(sizes) if sizes else 0.0
    return result


def _is_heading_by_text_rules(line: str, idx: int, all_lines: list[str]) -> bool:
    """Method B: text-pattern heading detection."""
    if not line or len(line) >= 120:
        return False
    if _is_junk_heading(line):           # reject math formulae and dot-filler table cells
        return False
    if line[-1] in ".,:;)":
        return False
    if _PURELY_NUMERIC_RE.match(line):
        return False
    if len(line) <= 4:
        return False
    if not re.search(r"[a-zA-Z]", line):
        return False

    # Next non-empty line must be strictly longer
    found_longer = False
    for j in range(idx + 1, min(idx + 6, len(all_lines))):
        nxt = all_lines[j].strip()
        if nxt:
            found_longer = len(nxt) > len(line)
            break
    if not found_longer:
        return False

    # Exclusions
    if line.startswith("("):
        return False
    lower = line.lower()
    if lower.startswith("note") or lower.startswith("provided"):
        return False
    if lower.startswith("table") or lower.startswith("figure"):
        return False
    if _SUBLIST_RE.match(line):
        return False
    if line.isupper() and len(line) < 5:
        return False
    if _BARE_SUBSEC_RE.match(line):
        return False

    # Bug 2: headings never start with a lowercase letter
    if line[0].islower():
        return False
    # Bug 2: a line ending with a preposition/conjunction is a mid-sentence cut
    if _TRAILING_PREP_RE.search(line):
        return False
    # Bug 2: body sentences contain verb phrases; skip unless it's a regulation heading
    if (_SENTENCE_VERB_RE.search(line)
            and not _STARTS_NUM_ROMAN_RE.match(line)
            and not _REGULATION_NUM_RE.match(line)):
        return False
    # Bug 2: definition entries ("iii. BoS shall mean...") are not headings
    if _DEFN_ENTRY_RE.match(line):
        return False

    return True


def detect_headings(pages: list[dict]) -> list[dict]:
    """
    Detect headings across all pages using Method A (font size) and
    Method B (text rules). A line is a heading if it passes either.
    """
    all_headings: list[dict] = []

    for page in pages:
        avg_font = page["avg_font_size"]
        page_num = page["page_number"]
        font_map = _build_line_font_map(page["chars"])

        raw_lines = page["raw_text"].split("\n")
        stripped  = [l.strip() for l in raw_lines]

        # Collect candidates for this page
        candidates: list[dict] = []
        for i, line in enumerate(stripped):
            if not line:
                continue

            is_heading = False

            # Positive fast-path: R.N(X).N regulation numbers are always headings
            if _REGULATION_NUM_RE.match(line) and not _is_junk_heading(line):
                is_heading = True

            if not is_heading:
                # Method A — font size (fix 1: threshold raised to +2.5)
                line_font = font_map.get(line, 0.0)
                if avg_font > 0 and line_font > avg_font + 2.5:
                    # Still reject math formulae / table fillers even when font is large
                    if not _is_junk_heading(line):
                        is_heading = True

            # Method B — text rules (fallback)
            if not is_heading:
                is_heading = _is_heading_by_text_rules(line, i, stripped)

            if is_heading:
                candidates.append({
                    "heading_text":  line,
                    "page_number":   page_num,
                    "line_position": i,
                })

        # Fix 2: keep only headings followed by ≥3 body lines before the next heading.
        # Regulation headings (R.N(X).N) are always kept even when their section is short.
        kept: list[dict] = []
        for j, h in enumerate(candidates):
            pos        = h["line_position"]
            next_pos   = candidates[j + 1]["line_position"] if j + 1 < len(candidates) else len(stripped)
            body_count = sum(1 for k in range(pos + 1, next_pos) if stripped[k])
            if body_count >= 3 or _REGULATION_NUM_RE.match(h["heading_text"]):
                kept.append(h)

        all_headings.extend(kept)

    return all_headings


# =============================================================================
# STEP 3 — Build section tree
# =============================================================================

def build_sections(pages: list[dict], headings: list[dict]) -> list[dict]:
    """
    Group all page content into sections delimited by detected headings.
    Fallback: one section per page if no headings found.
    """
    if not pages:
        return []

    if not headings:
        return [
            {
                "heading":       f"Page {p['page_number']}",
                "page_number":   p["page_number"],
                "content_lines": [l.strip() for l in p["raw_text"].split("\n") if l.strip()],
                "raw_text":      p["raw_text"].strip(),
            }
            for p in pages
        ]

    # Build lookup: (page_number, line_position) → heading_text
    heading_lookup: dict[tuple[int, int], str] = {
        (h["page_number"], h["line_position"]): h["heading_text"]
        for h in headings
    }

    sections:      list[dict] = []
    cur_heading    = "Preamble"
    cur_page       = pages[0]["page_number"]
    cur_lines:     list[str] = []

    for page in pages:
        page_num  = page["page_number"]
        raw_lines = page["raw_text"].split("\n")

        for line_pos, raw_line in enumerate(raw_lines):
            stripped = raw_line.strip()
            key = (page_num, line_pos)

            if key in heading_lookup:
                # Flush current section
                body = "\n".join(cur_lines).strip()
                if body:
                    sections.append({
                        "heading":       cur_heading,
                        "page_number":   cur_page,
                        "content_lines": list(cur_lines),
                        "raw_text":      body,
                    })
                cur_heading = heading_lookup[key]
                cur_page    = page_num
                cur_lines   = []
            else:
                if stripped:
                    cur_lines.append(stripped)

    # Flush final section
    body = "\n".join(cur_lines).strip()
    if body:
        sections.append({
            "heading":       cur_heading,
            "page_number":   cur_page,
            "content_lines": list(cur_lines),
            "raw_text":      body,
        })

    return sections


# =============================================================================
# STEP 4 — Chunk within sections
# =============================================================================

def _find_table_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs for table blocks (end is exclusive).
    Detects pipe-based tables (2+ '|' chars) and column-aligned tables
    (3+ consecutive lines with 3+ space gap between columns).
    """
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        # Pipe table
        if lines[i].count("|") >= 2:
            start = i
            while i < len(lines) and lines[i].count("|") >= 2:
                i += 1
            blocks.append((start, i))
            continue

        # Column-aligned table
        j = i
        while j < len(lines) and _COL_ALIGN_RE.search(lines[j]):
            j += 1
        if j - i >= 3:
            blocks.append((i, j))
            i = j
            continue

        i += 1
    return blocks


def _build_overlap(text: str) -> str:
    """
    Return the last ~100 tokens of *text*, trimmed back to a sentence start.
    Hard cap: MAX_OVERLAP_TOKENS.
    """
    tokens = enc.encode(text)
    if len(tokens) <= OVERLAP_TARGET:
        return text

    overlap_toks = tokens[-MAX_OVERLAP_TOKENS:]
    try:
        overlap_text = enc.decode(overlap_toks)
    except Exception:
        return ""

    # Trim to nearest sentence start (first capital letter after ". ")
    match = re.search(r"(?<=\.\s)[A-Z(]", overlap_text)
    if match:
        overlap_text = overlap_text[match.start():]

    # Hard cap after trimming
    final_toks = enc.encode(overlap_text)
    if len(final_toks) > MAX_OVERLAP_TOKENS:
        try:
            overlap_text = enc.decode(final_toks[-OVERLAP_TARGET:])
        except Exception:
            overlap_text = ""

    return overlap_text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries, skipping periods inside known abbreviations."""
    protected = _ABBREVS_RE.sub(lambda m: m.group().replace(".", "<<DOT>>"), text)
    parts = re.split(r"(?<=\.)\s+", protected)
    return [p.replace("<<DOT>>", ".").strip() for p in parts if p.strip()]


def _split_into_chunks(text: str) -> list[str]:
    """
    Split *text* into ≤MAX_TOKENS chunks at sentence boundaries ('. ' or '.\n').
    Consecutive chunks carry ~100-token overlap from the previous chunk.
    """
    if count_tokens(text) <= MAX_TOKENS:
        return [text]

    sentences = _split_sentences(text.strip())

    if not sentences:
        return [text]

    # First pass: group into raw chunks (no overlap) at MAX_TOKENS boundary
    raw_groups: list[list[str]] = []
    current: list[str] = []
    current_tok = 0

    for s in sentences:
        s_tok = count_tokens(s)
        if current and current_tok + s_tok + 1 > MAX_TOKENS:
            raw_groups.append(current[:])
            current = [s]
            current_tok = s_tok
        else:
            current.append(s)
            current_tok += s_tok + 1
    if current:
        raw_groups.append(current)

    # Second pass: prepend overlap from previous group
    result: list[str] = []
    for i, group in enumerate(raw_groups):
        chunk_text = " ".join(group)
        if i > 0:
            overlap = _build_overlap(" ".join(raw_groups[i - 1]))
            if overlap:
                chunk_text = overlap + " " + chunk_text
        result.append(chunk_text)

    # Bug 3 Fix B: merge any non-first chunk under MIN_TOKENS into the previous chunk
    merged: list[str] = []
    for chunk_text in result:
        if merged and count_tokens(chunk_text) < MIN_TOKENS:
            merged[-1] = merged[-1] + " " + chunk_text
        else:
            merged.append(chunk_text)
    return merged


def _keep_section(section: dict) -> bool:
    """Fix 4: discard thin sections unless the heading contains a policy keyword."""
    if count_tokens(section["raw_text"]) >= MIN_SECTION_TOKENS:
        return True
    return bool(_POLICY_KEYWORDS_RE.search(section["heading"]))


def chunk_section(
    section:        dict,
    source_url:     str,
    doc_title:      str,
    doc_type:       str,
    date_published: str,
    date_scraped:   str,
    chunk_counter:  list[int],
) -> tuple[list[dict], int]:
    """
    Produce chunk records for a single section.
    Returns (chunks, table_chunk_count).
    chunk_counter is a mutable [n] for globally sequential IDs.
    """
    heading  = section["heading"]
    page_num = section["page_number"]
    lines    = section["content_lines"]
    raw_text = section["raw_text"]

    table_count = 0

    def make_chunk(content: str, chunk_index: int) -> dict:
        full_text = f"Section: {heading}\n\n{content}"
        n = chunk_counter[0]
        chunk_counter[0] += 1
        return {
            "chunk_id":        f"auto_{n:03d}",
            "text":            full_text,
            "source_url":      source_url,
            "document_title":  doc_title,
            "section_heading": heading,
            "document_type":   doc_type,
            "date_published":  date_published,
            "date_scraped":    date_scraped,
            "chunk_index":     chunk_index,
            "token_count":     count_tokens(full_text),
            "is_ocr":          False,
            "page_number":     page_num,
        }

    full_tok = count_tokens(raw_text)

    # Rule 3 — too small: keep as standalone, never merge
    if full_tok < MIN_TOKENS:
        log.info(f"Small section kept standalone: {heading}")
        return [make_chunk(raw_text, 0)], 0

    # Rule 1 — fits in one chunk (primary case)
    if full_tok <= MAX_TOKENS:
        return [make_chunk(raw_text, 0)], 0

    # Rule 2 / Table — section too large, needs splitting
    table_blocks = {start: end for start, end in _find_table_blocks(lines)}

    chunks:           list[dict] = []
    chunk_index       = 0
    pending:          list[str] = []
    i                 = 0

    def _flush_pending() -> None:
        nonlocal chunk_index, pending
        if not pending:
            return
        blob = "\n".join(pending)
        tok  = count_tokens(blob)
        if tok < MIN_TOKENS:
            log.info(f"Small section kept standalone: {heading}")
            chunks.append(make_chunk(blob, chunk_index))
            chunk_index += 1
        elif tok <= MAX_TOKENS:
            chunks.append(make_chunk(blob, chunk_index))
            chunk_index += 1
        else:
            for sub in _split_into_chunks(blob):
                chunks.append(make_chunk(sub, chunk_index))
                chunk_index += 1
        pending = []

    while i < len(lines):
        if i in table_blocks:
            end = table_blocks[i]
            _flush_pending()

            table_lines = lines[i:end]
            table_text  = "\n".join(table_lines)
            table_tok   = count_tokens(table_text)

            # Rule 4 — table handling
            if table_tok <= TABLE_MAX_TOKENS:
                chunks.append(make_chunk(table_text, chunk_index))
                chunk_index += 1
                table_count += 1
            else:
                header    = table_lines[0] if table_lines else ""
                data_rows = table_lines[1:] if len(table_lines) > 1 else table_lines
                for g in range(0, len(data_rows), 6):
                    group = data_rows[g:g + 6]
                    group_text = (
                        f"[Headers: {header}]\n" + "\n".join(group)
                        if header else "\n".join(group)
                    )
                    chunks.append(make_chunk(group_text, chunk_index))
                    chunk_index += 1
                    table_count += 1

            i = end
        else:
            pending.append(lines[i])
            i += 1

    _flush_pending()
    return chunks, table_count


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DTU PDF Chunker")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument("--url",      required=True, help="Source URL")
    parser.add_argument("--title",    required=True, help="Document title")
    parser.add_argument("--doc-type", required=True, dest="doc_type", help="Document type")
    parser.add_argument("--date",     required=True, help="Publication date (YYYY-MM-DD)")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not pdf_path.exists():
        log.error(f"File not found: {pdf_path}")
        sys.exit(1)

    date_scraped = date.today().isoformat()
    stem         = pdf_path.stem

    output_dir  = Path("data/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_chunks.jsonl"

    # Step 1
    log.info("Step 1 — Extracting pages...")
    pages, skipped, total_pages, class_counts = extract_pages(pdf_path)
    processed = len(pages)

    # Step 2
    log.info("Step 2 — Detecting headings...")
    headings = detect_headings(pages)

    # Step 3
    log.info("Step 3 — Building section tree...")
    sections = build_sections(pages, headings)

    # Fix 4: discard sections below MIN_SECTION_TOKENS unless policy-keyword heading
    n_before = len(sections)
    sections = [s for s in sections if _keep_section(s)]
    log.info(f"  Sections after content filter: {len(sections)} kept, {n_before - len(sections)} discarded")

    # Step 4 + 5
    log.info("Step 4 — Chunking sections...")
    all_chunks:    list[dict] = []
    chunk_counter: list[int]  = [1]
    total_table   = 0

    for section in sections:
        chunks, tc = chunk_section(
            section        = section,
            source_url     = args.url,
            doc_title      = args.title,
            doc_type       = args.doc_type,
            date_published = args.date,
            date_scraped   = date_scraped,
            chunk_counter  = chunk_counter,
        )
        all_chunks.extend(chunks)
        total_table += tc

    # SHA-256 content deduplication
    seen_hashes: set[str] = set()
    final_chunks: list[dict] = []
    dedup_count = 0
    for chunk in all_chunks:
        h = hashlib.sha256(chunk["text"].encode()).hexdigest()
        if h in seen_hashes:
            log.info(
                f"Duplicate skipped: {chunk['section_heading']} "
                f"(page {chunk['page_number']})"
            )
            dedup_count += 1
            continue
        seen_hashes.add(h)
        final_chunks.append(chunk)

    # Renumber chunk_id and chunk_index after dedup so there are no gaps
    section_idx_map: dict[tuple, int] = {}
    for i, chunk in enumerate(final_chunks, start=1):
        chunk["chunk_id"] = f"auto_{i:03d}"
        key = (chunk["source_url"], chunk["section_heading"], chunk["page_number"])
        idx = section_idx_map.get(key, 0)
        chunk["chunk_index"] = idx
        section_idx_map[key] = idx + 1
        chunk["token_count"] = count_tokens(chunk["text"])

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in final_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Summary (stats over the deduplicated final set)
    token_counts = [c["token_count"] for c in final_chunks]
    n = len(token_counts)
    avg_tok  = int(sum(token_counts) / n) if n else 0
    min_tok  = min(token_counts)          if n else 0
    max_tok  = max(token_counts)          if n else 0
    over_600 = sum(1 for t in token_counts if t > 600)
    under_100 = sum(1 for t in token_counts if t < 100)

    print("\n================================")
    print("CHUNKING COMPLETE")
    print("================================")
    print(f"Input file     : {pdf_path.name}")
    print(f"Total pages    : {total_pages}")
    print(f"Pages skipped  : {len(skipped)} (scanned/image)")
    print(f"Pages processed: {processed}")
    print()
    print(f"Sections detected  : {len(sections)}")
    print(f"Chunks created     : {n}")
    print(f"Avg chunk size     : {avg_tok} tokens")
    print(f"Min chunk size     : {min_tok} tokens")
    print(f"Max chunk size     : {max_tok} tokens")
    print(f"Chunks > 600 tokens: {over_600} (were split)")
    print(f"Chunks < 100 tokens: {under_100} (kept standalone)")
    print(f"Table chunks       : {total_table}")
    print(f"Duplicates skipped : {dedup_count}")
    print()
    print(f"Classification breakdown:")
    print(f"  Pages classified policy  : {class_counts.get('policy',   0)}")
    print(f"  Pages classified notice  : {class_counts.get('notice',   0)}")
    print(f"  Pages classified syllabus: {class_counts.get('syllabus', 0)} (skipped)")
    print(f"  Pages classified contact : {class_counts.get('contact',  0)} (skipped)")
    print(f"  Pages classified form    : {class_counts.get('form',     0)} (skipped)")
    print(f"  Pages classified skip    : {class_counts.get('skip',     0)} (skipped)")
    print()
    print(f"Output written to: {output_path}")
    print()
    print(f"Validate with:")
    print(f"python ingestion/test_chunker.py \\")
    print(f"  data/chunks/{stem}_chunks.jsonl")


if __name__ == "__main__":
    main()
