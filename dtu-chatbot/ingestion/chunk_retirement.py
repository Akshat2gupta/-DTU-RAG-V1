"""
Retire raw chunks that compete with curated content or duplicate each other.

Two rules:

RULE 1 — superseded regulations: ordinance chunks whose section is an
  R. 1(B).x regulation (or the 2017-18 'REGULATIONS' equivalent) are fully
  covered by data/curated/academic_rules.md. With 8 ordinance years indexed,
  every academic query competed against ~8 near-identical copies of these.
  Exception: the unfair-means PENALTY annexure chunks are kept (penalty
  specifics are not in the curated doc).

RULE 2 — duplicated front-matter: non-regulation, non-table ordinance
  sections that appear under the same heading in several ordinance years
  (About University, Computer Centre, DTU Times, Preamble, ...) are kept only
  for the NEWEST year; older copies are deleted.

RULE 3 — repeated-heading floods: when one document yields more than
  _FLOOD_LIMIT chunks under the same section heading (e.g. 488 table chunks
  all titled "Code Title Percentage of attendance" from a misfiring heading
  detector on syllabus pages), only the lowest-chunk_index few are kept.
  Legitimate sections never repeat a heading dozens of times.

RULE 4 — ordinance prose wholesale: ALL non-table machine chunks from
  ordinance PDFs are retired. The curated docs (academic_rules,
  ordinance_old_batches, admissions, hostel, placement_stats) cover the
  rules, fee facts, recruiters and old-batch differences; remaining ordinance
  prose is front-matter/marketing that conflicts with curated figures and
  poisons answers. Syllabus/scheme TABLES are kept (block_type == 'table').
  Exception: unfair-means penalty annexure prose is kept (not curated).

This module is called AUTOMATICALLY at the end of batch_index.py and
ocr_worker.py runs, so re-indexing cannot resurrect retired chunks. The CLI
wrapper (vertical_slice/retire_chunks.py) exists for manual runs/dry-runs.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

COLLECTION = "dtu_rag"

_REG_HEADING_RE = re.compile(r"R\.?\s*1?\s*\(B\)\.?\s*\d+|^REGULATIONS\b", re.IGNORECASE)
_REG_TEXT_RE    = re.compile(r"R\.\s*1\s*\(B\)\.\s*\d+\s+[A-Z]")
_KEEP_RE        = re.compile(r"unfair\s*means", re.IGNORECASE)   # penalty annexure stays

_YEAR_RE = re.compile(r"(20\d{2})")

_FLOOD_LIMIT = 15   # > this many chunks under one heading in one doc = misparse
_FLOOD_KEEP  = 3    # how many lowest-index chunks survive from a flood


def _ordinance_year(url: str) -> int:
    m = _YEAR_RE.search(url)
    return int(m.group(1)) if m else 0


def find_retired(client: QdrantClient, collection: str = COLLECTION):
    """Return (rule1_points, rule2_points, rule3_points) to be deleted."""
    pts, off = [], None
    while True:
        batch, off = client.scroll(collection, limit=500, offset=off,
                                   with_payload=True, with_vectors=False)
        pts += batch
        if off is None:
            break

    ordinance = [
        p for p in pts
        if "ordinance" in (p.payload.get("source_url") or "").lower()
        and p.payload.get("document_type") != "curated_stats"
    ]

    # RULE 1 — regulation chunks superseded by curated academic_rules.md
    rule1 = []
    for p in ordinance:
        heading = p.payload.get("section_heading") or ""
        text    = p.payload.get("text") or ""
        if _KEEP_RE.search(heading):
            continue
        if _REG_HEADING_RE.search(heading) or _REG_TEXT_RE.search(text[:400]):
            rule1.append(p)
    rule1_ids = {p.id for p in rule1}

    # RULE 2 — same front-matter heading across ordinance years: keep newest
    groups: dict[str, list] = defaultdict(list)
    for p in ordinance:
        if p.id in rule1_ids:
            continue
        if p.payload.get("block_type") == "table":
            continue   # tables carry year-specific numbers — keep all years
        heading = (p.payload.get("section_heading") or "").strip().lower()
        heading = re.sub(r"\s+", " ", heading)
        if len(heading) < 4:
            continue
        groups[heading].append(p)

    rule2 = []
    for heading, plist in groups.items():
        years = {_ordinance_year(p.payload.get("source_url") or "") for p in plist}
        if len(years) < 2:
            continue
        newest = max(years)
        rule2.extend(p for p in plist
                     if _ordinance_year(p.payload.get("source_url") or "") != newest)
    rule2_ids = {p.id for p in rule2}

    # RULE 3 — repeated-heading floods within a single ORDINANCE document
    # (misfiring heading detector on syllabus pages, e.g. 488 chunks titled
    # "Code Title Percentage of attendance"). Scoped to ordinances only:
    # other documents repeat headings legitimately (faculty pages repeat
    # "Assistant Professor" once per professor).
    flood_groups: dict[tuple, list] = defaultdict(list)
    for p in ordinance:
        if p.id in rule1_ids or p.id in rule2_ids:
            continue
        heading = (p.payload.get("section_heading") or "").strip().lower()
        if len(heading) < 4:
            continue
        flood_groups[(p.payload.get("source_url"), heading)].append(p)

    rule3 = []
    for key, plist in flood_groups.items():
        if len(plist) <= _FLOOD_LIMIT:
            continue
        plist.sort(key=lambda p: int(p.payload.get("chunk_index") or 0))
        rule3.extend(plist[_FLOOD_KEEP:])
    rule3_ids = {p.id for p in rule3}

    # RULE 4 — all remaining ordinance prose (curated docs supersede it;
    # tables stay for year-specific schemes/fees; unfair-means annexure stays)
    rule4 = []
    for p in ordinance:
        if p.id in rule1_ids or p.id in rule2_ids or p.id in rule3_ids:
            continue
        if p.payload.get("block_type") == "table":
            continue
        heading = p.payload.get("section_heading") or ""
        if _KEEP_RE.search(heading):
            continue
        rule4.append(p)

    return rule1, rule2, rule3, rule4


def retire_chunks(client: QdrantClient, collection: str = COLLECTION,
                  dry_run: bool = False, verbose: bool = True) -> int:
    """Delete superseded/duplicate chunks. Returns the number deleted.

    Safe to call after every indexing run — a no-op when nothing matches."""
    rule1, rule2, rule3, rule4 = find_retired(client, collection)
    ids = list({p.id for p in rule1} | {p.id for p in rule2}
               | {p.id for p in rule3} | {p.id for p in rule4})

    if verbose and ids:
        print(f"Chunk retirement: {len(rule1)} superseded regulation chunks, "
              f"{len(rule2)} duplicated front-matter chunks, "
              f"{len(rule3)} repeated-heading flood chunks, "
              f"{len(rule4)} ordinance prose chunks"
              + (" (dry run)" if dry_run else ""))
    if dry_run or not ids:
        return 0

    for i in range(0, len(ids), 500):
        client.delete(collection, points_selector=PointIdsList(points=ids[i:i + 500]))
    time.sleep(1)
    if verbose:
        print(f"Chunk retirement: deleted {len(ids)} chunks "
              f"(collection now {client.get_collection(collection).points_count} points)")
    return len(ids)
