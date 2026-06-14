"""
Curated-source registry.

Documents in dtu-chatbot/data/curated/*.md are hand-verified replacements for
sources that machine extraction mangles (e.g. scanned placement-stats sheets).

Two kinds of URL lines in curated docs:
- "Source: <url>"     — citation only; sets the chunk's source_url payload.
- "Supersedes: <url>" — the curated doc REPLACES that raw document entirely;
  indexers (batch_index, ocr_worker) must skip it or a re-run would re-add
  garbled machine-extracted chunks alongside the clean curated ones.

A curated doc that merely summarizes part of a large document (e.g. ordinance
rules) should cite it with Source: but must NOT list it under Supersedes:.
"""
from __future__ import annotations

import re
from pathlib import Path

_CURATED_DIR    = Path(__file__).resolve().parent.parent / "data" / "curated"
_SUPERSEDES_RE  = re.compile(r"^Supersedes:\s*(\S+)", re.MULTILINE)


def curated_source_urls() -> set[str]:
    """URLs whose content is fully superseded by a curated document."""
    urls: set[str] = set()
    if not _CURATED_DIR.exists():
        return urls
    for md in _CURATED_DIR.glob("*.md"):
        urls.update(_SUPERSEDES_RE.findall(md.read_text(encoding="utf-8")))
    return urls
