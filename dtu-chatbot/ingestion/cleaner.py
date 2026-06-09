"""
Stage 4 — HTML clean worker.

Reads every document with clean_status='pending' from the manifest (these are
HTML docs — PDFs get clean_status='skipped' from schema defaults), parses each
saved HTML body into a Document IR, applies the normalization pipeline below,
serialises the cleaned IR to data/clean/{url_hash}.json, and updates the
manifest atomically.

Cleaning pipeline (applied to every block):
    1. Unicode normalisation (NFKC + explicit char map)
       – Ligatures, curly quotes, en/em dashes, non-breaking spaces, zero-width
         chars, BOMs, fullwidth characters.
    2. Whitespace collapse (post-NFKC sweep).
    3. Boilerplate paragraph removal
       – Nav breadcrumbs that slipped past the HTML parser's noise filter.
       – Bare URLs and single-word fragments masquerading as paragraphs.
    4. Trivial block removal
       – Paragraphs / headings below a configurable character minimum.
       – Heading text that is purely numeric or punctuation.
    5. Within-document exact deduplication
       – First occurrence wins; later identical blocks are dropped.
       – Tables are always kept (duplicate table rows matter structurally).

Usage
-----
    cd dtu-chatbot/
    python ingestion/cleaner.py
    python ingestion/cleaner.py --concurrency 8 --log-level DEBUG
    python ingestion/cleaner.py --manifest-path path/to/manifest.db \\
                                --raw-html-dir data/raw/html \\
                                --clean-dir data/clean
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ── path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.document_ir import Block, Document, Heading, ListBlock, Paragraph, Table
from ingestion.html_parser import parse_html_file
from manifest.manifest import ManifestDB
from manifest.resumability import recover_and_get_pending

log = logging.getLogger(__name__)

STAGE: Final[str] = "clean"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CleanConfig:
    """All tunables in one place."""
    manifest_path: Path  = Path("manifest.db")
    raw_html_dir: Path   = Path("data/raw/html")
    clean_dir: Path      = Path("data/clean")
    concurrency: int     = 8     # ThreadPoolExecutor workers for lxml parsing
    min_para_chars: int  = 20    # drop paragraphs shorter than this
    min_heading_chars: int = 2   # drop headings shorter than this


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class CleanError(Exception):
    """Base class for cleaner errors."""


class HtmlBodyNotFoundError(CleanError):
    """Raw HTML file does not exist on disk."""


class ParseFailedError(CleanError):
    """html_parser raised an unexpected exception."""


# ─────────────────────────────────────────────────────────────────────────────
# Unicode & whitespace normalisation
# ─────────────────────────────────────────────────────────────────────────────

# Characters that NFKC does not fully handle but we always want normalised.
_CHAR_MAP: Final[dict[str, str]] = {
    "‘": "'",    # left single quotation mark
    "’": "'",    # right single quotation mark
    "“": '"',    # left double quotation mark
    "”": '"',    # right double quotation mark
    "–": " - ",  # en dash
    "—": " - ",  # em dash
    "…": "...",  # horizontal ellipsis
    " ": " ",    # non-breaking space
    "­": "",     # soft hyphen
    "​": "",     # zero-width space
    "‌": "",     # zero-width non-joiner
    "‍": "",     # zero-width joiner
    "﻿": "",     # byte order mark
    "•": "-",    # bullet (•) → hyphen so text stays readable
    "●": "-",    # black circle bullet
    "◦": "-",    # white bullet
}

_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """
    NFKC → char map → whitespace collapse.

    NFKC resolves ligatures (ﬁ→fi), fullwidth ASCII (Ａ→A), circled digits, and
    superscript numbers.  The explicit char map handles typographic punctuation
    that NFKC leaves as-is.
    """
    text = unicodedata.normalize("NFKC", text)
    for char, replacement in _CHAR_MAP.items():
        if char in text:                 # avoid unnecessary replace on clean text
            text = text.replace(char, replacement)
    return _WS_RE.sub(" ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Boilerplate detection
# ─────────────────────────────────────────────────────────────────────────────

# "Home > Academics > Scholarships" / "Home | About | Contact" navigation
# breadcrumbs that slipped past the HTML parser's noise filter.
_BREADCRUMB_RE: Final[re.Pattern[str]] = re.compile(
    r"^[\w\s.&'\-()]+(?:\s*[>|/»·]\s*[\w\s.&'\-()]+){2,}$"
)

# Bare http(s) URL with optional trailing whitespace — no useful prose.
_BARE_URL_RE: Final[re.Pattern[str]] = re.compile(r"^https?://\S+$")

# Purely numeric / punctuation content masquerading as a paragraph.
_JUNK_CONTENT_RE: Final[re.Pattern[str]] = re.compile(r"^[\d\s.,;:\-–—|/\\()[\]{}]+$")


def _is_boilerplate_paragraph(text: str) -> bool:
    """
    True for text that carries no answerable content.

    Covers:
    - Nav breadcrumbs: "Home > Academics > Ordinance"
    - Bare URLs
    - Strings of only numbers and punctuation (page numbers, rule counters)
    - Extremely short fragments (single word, isolated digit)
    """
    stripped = text.strip()
    if not stripped:
        return True
    if _BARE_URL_RE.match(stripped):
        return True
    if _JUNK_CONTENT_RE.match(stripped):
        return True
    if _BREADCRUMB_RE.match(stripped) and len(stripped) < 250:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Block-level cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean_block(block: Block) -> Block | None:
    """
    Normalise a single block in place and return it, or None to drop it.

    Tables are always kept — even a short table is structured data worth
    preserving.  ListBlock items are normalised individually; the block is
    dropped if every item becomes empty.
    """
    if isinstance(block, Paragraph):
        text = _normalize_text(block.text)
        if not text:
            return None
        return Paragraph(text=text, page=block.page)

    if isinstance(block, Heading):
        text = _normalize_text(block.text)
        if not text:
            return None
        return Heading(text=text, level=block.level, page=block.page)

    if isinstance(block, ListBlock):
        items = [_normalize_text(item) for item in block.items]
        items = [i for i in items if i]
        if not items:
            return None
        return ListBlock(items=items, ordered=block.ordered, page=block.page)

    if isinstance(block, Table):
        headers = [_normalize_text(h) for h in block.headers]
        rows = [
            [_normalize_text(cell) for cell in row]
            for row in block.rows
        ]
        rows = [r for r in rows if any(r)]   # drop fully-empty rows
        caption = _normalize_text(block.caption) if block.caption else None
        if not rows:
            return None
        return Table(rows=rows, headers=headers, caption=caption or None, page=block.page)

    return block   # unknown block type: pass through


def _is_trivial(block: Block, cfg: CleanConfig) -> bool:
    """
    True for blocks too short or content-free to be worth chunking.

    Headings are held to a lower bar (min_heading_chars) so short section
    labels like "FAQ" or "T&P" are kept for breadcrumb purposes.
    """
    if isinstance(block, Paragraph):
        return (
            len(block.text) < cfg.min_para_chars
            or _is_boilerplate_paragraph(block.text)
        )
    if isinstance(block, Heading):
        return len(block.text) < cfg.min_heading_chars
    if isinstance(block, ListBlock):
        # A list is non-trivial if at least one item meets the para threshold.
        return not any(len(i) >= cfg.min_para_chars for i in block.items)
    return False   # tables are never trivial


# ─────────────────────────────────────────────────────────────────────────────
# Within-document deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _block_fingerprint(block: Block) -> str | None:
    """
    A lowercase, whitespace-collapsed key for deduplication.

    Returns None for Tables — structural duplicates in tables are rare and
    could represent legitimately repeated data (e.g. a fee appearing in two
    sections), so they are never deduplicated.
    """
    if isinstance(block, Heading):
        return f"h:{block.text.lower()}"
    if isinstance(block, Paragraph):
        return f"p:{block.text.lower()}"
    if isinstance(block, ListBlock):
        joined = " | ".join(item.lower() for item in block.items)
        return f"l:{joined}"
    return None   # Table: never deduplicated


def _dedup_blocks(blocks: list[Block]) -> list[Block]:
    """
    Remove exact duplicates within the document (first occurrence wins).

    Case-insensitive: a repeated paragraph differing only in capitalisation is
    still a duplicate.  Tables are always preserved (no deduplication).
    """
    seen: set[str] = set()
    result: list[Block] = []
    for block in blocks:
        fp = _block_fingerprint(block)
        if fp is None:
            result.append(block)   # Table: keep unconditionally
            continue
        if fp not in seen:
            seen.add(fp)
            result.append(block)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full document cleaning pass
# ─────────────────────────────────────────────────────────────────────────────

def clean_document(doc: Document, cfg: CleanConfig) -> Document:
    """
    Run all cleaning passes over a Document IR.

    Order matters:
        normalise → filter trivial → deduplicate

    Normalisation first so deduplication keys are canonical.
    """
    # Pass 1: unicode + whitespace normalisation
    normalised: list[Block] = []
    for block in doc.blocks:
        cleaned = _clean_block(block)
        if cleaned is not None:
            normalised.append(cleaned)

    # Pass 2: drop trivial / boilerplate blocks
    filtered = [b for b in normalised if not _is_trivial(b, cfg)]

    # Pass 3: within-document exact deduplication
    deduped = _dedup_blocks(filtered)

    return Document(
        url=doc.url,
        title=_normalize_text(doc.title),
        source_format=doc.source_format,
        doc_type=doc.doc_type,
        date_published=doc.date_published,
        blocks=deduped,
    )


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _raw_html_path(raw_html_dir: Path, url: str) -> Path:
    """Derive the saved HTML body path from the source URL."""
    return raw_html_dir / f"{_url_hash(url)}.html"


def _clean_json_path(clean_dir: Path, url: str) -> Path:
    """Derive the output clean-IR JSON path from the source URL."""
    return clean_dir / f"{_url_hash(url)}.json"


def _save_clean_doc(doc: Document, clean_dir: Path) -> Path:
    """Serialise a cleaned Document IR to JSON and return the written path."""
    dest = _clean_json_path(clean_dir, doc.url)
    dest.write_text(json.dumps(doc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Parse + clean worker (runs in a thread — lxml releases the GIL)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_and_clean(
    url: str,
    doc_type: str,
    title: str | None,
    date_published: str | None,
    raw_html_dir: Path,
    clean_dir: Path,
    cfg: CleanConfig,
) -> Path:
    """
    Parse the saved HTML file, clean the IR, and save the result.

    Returns the path to the written JSON file.
    Raises CleanError (or a subclass) on any recoverable failure.
    """
    raw_path = _raw_html_path(raw_html_dir, url)
    if not raw_path.exists():
        raise HtmlBodyNotFoundError(f"HTML body not found: {raw_path}")

    try:
        doc = parse_html_file(
            raw_path,
            url=url,
            doc_type=doc_type,
            title=title or None,
            date_published=date_published or None,
        )
    except Exception as exc:
        raise ParseFailedError(f"html_parser raised: {exc}") from exc

    cleaned = clean_document(doc, cfg)
    return _save_clean_doc(cleaned, clean_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: CleanConfig) -> None:
    """Full clean pass: recover → fetch pending → parse+clean all → summarise."""
    cfg.clean_dir.mkdir(parents=True, exist_ok=True)

    with ManifestDB(cfg.manifest_path) as db:
        pending = recover_and_get_pending(db, STAGE)
        total = len(pending)

        if total == 0:
            log.info("No documents pending clean — nothing to do.")
            return

        log.info(
            "Starting clean pass: %d documents, concurrency=%d",
            total, cfg.concurrency,
            extra={"total": total, "concurrency": cfg.concurrency},
        )

        done = failed = 0

        # Mark all pending rows as running before we hand work to threads.
        # This way a crash mid-batch is recoverable by recover_and_get_pending.
        for doc in pending:
            db.mark_running(doc["url"], STAGE)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=cfg.concurrency, thread_name_prefix="cleaner"
        ) as executor:
            future_to_url = {
                executor.submit(
                    _parse_and_clean,
                    doc["url"],
                    doc["document_type"] or "unknown",
                    doc["title"],
                    doc["date_published"],
                    cfg.raw_html_dir,
                    cfg.clean_dir,
                    cfg,
                ): doc["url"]
                for doc in pending
            }

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    clean_path = future.result()
                    db.update_stage(url, STAGE, "done", notes=str(clean_path))
                    done += 1
                    log.info(
                        "OK  %s → %s",
                        url, clean_path.name,
                        extra={"url": url, "outcome": "ok", "path": str(clean_path)},
                    )
                except HtmlBodyNotFoundError as exc:
                    db.update_stage(url, STAGE, "failed", notes=str(exc))
                    failed += 1
                    log.warning(
                        "FAILED (missing)  %s | %s",
                        url, exc,
                        extra={"url": url, "outcome": "failed_missing"},
                    )
                except ParseFailedError as exc:
                    db.update_stage(url, STAGE, "failed", notes=str(exc))
                    failed += 1
                    log.error(
                        "FAILED (parse)  %s | %s",
                        url, exc,
                        extra={"url": url, "outcome": "failed_parse"},
                    )
                except Exception as exc:
                    db.update_stage(url, STAGE, "failed", notes=f"unexpected: {exc}")
                    failed += 1
                    log.exception(
                        "FAILED (unexpected)  %s",
                        url,
                        extra={"url": url, "outcome": "failed_unexpected"},
                    )

        log.info(
            "Clean pass complete | done=%d failed=%d total=%d",
            done, failed, total,
            extra={"done": done, "failed": failed, "total": total},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone inspection helper
# ─────────────────────────────────────────────────────────────────────────────

def _inspect(html_path: Path, url: str, doc_type: str) -> None:
    """Print a before/after summary of what cleaning does to one HTML file."""
    cfg = CleanConfig()

    raw_doc = parse_html_file(html_path, url=url, doc_type=doc_type)
    clean_doc = clean_document(raw_doc, cfg)

    raw_kinds: dict[str, int] = {}
    for b in raw_doc.blocks:
        raw_kinds[b.kind] = raw_kinds.get(b.kind, 0) + 1

    clean_kinds: dict[str, int] = {}
    for b in clean_doc.blocks:
        clean_kinds[b.kind] = clean_kinds.get(b.kind, 0) + 1

    print(f"URL        : {url}")
    print(f"Title      : {clean_doc.title}")
    print(f"Raw blocks : {len(raw_doc.blocks)}  {raw_kinds}")
    print(f"Clean blocks: {len(clean_doc.blocks)}  {clean_kinds}")
    print(f"Removed    : {len(raw_doc.blocks) - len(clean_doc.blocks)} blocks")
    print()
    print("--- First 10 cleaned blocks ---")
    for b in clean_doc.blocks[:10]:
        if isinstance(b, Paragraph):
            preview = b.text[:120]
        elif isinstance(b, Heading):
            preview = f"[H{b.level}] {b.text}"
        elif isinstance(b, ListBlock):
            preview = b.as_text().replace("\n", " | ")[:120]
        else:
            preview = b.as_text()[:120]
        print(f"  [{b.kind:9s}] {preview}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> tuple[str, CleanConfig]:
    parser = argparse.ArgumentParser(
        description="Stage 4 — parse and clean HTML docs into Document IR JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- run subcommand (default) ---
    run_p = sub.add_parser("run", help="Process all pending HTML docs (default)")
    run_p.add_argument("--manifest-path", type=Path, default=Path("manifest.db"))
    run_p.add_argument("--raw-html-dir",  type=Path, default=Path("data/raw/html"))
    run_p.add_argument("--clean-dir",     type=Path, default=Path("data/clean"))
    run_p.add_argument("--concurrency",   type=int,  default=8)
    run_p.add_argument("--min-para-chars",    type=int, default=20)
    run_p.add_argument("--min-heading-chars", type=int, default=2)
    run_p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # --- inspect subcommand ---
    ins_p = sub.add_parser("inspect", help="Show before/after for a single HTML file")
    ins_p.add_argument("html_path", type=Path)
    ins_p.add_argument("--url", required=True)
    ins_p.add_argument("--doc-type", default="unknown")

    args = parser.parse_args()

    # If no subcommand given, default to "run"
    if args.command is None:
        args.command = "run"
        # re-parse to pick up defaults for the run subparser
        args = run_p.parse_args(sys.argv[1:])
        args.command = "run"

    return args.command, args


if __name__ == "__main__":
    command, args = _parse_args()

    if command == "inspect":
        _inspect(args.html_path, args.url, args.doc_type)
        sys.exit(0)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = CleanConfig(
        manifest_path=Path(args.manifest_path),
        raw_html_dir=Path(args.raw_html_dir),
        clean_dir=Path(args.clean_dir),
        concurrency=args.concurrency,
        min_para_chars=args.min_para_chars,
        min_heading_chars=args.min_heading_chars,
    )
    run(cfg)
