#!/usr/bin/env python3
"""
Chunk quality audit — scrolls every point in the Qdrant collection and flags:

  1. interleaved   : PDF two-column interleaving (long sentences that never end,
                     mid-word jumps) — detected via low sentence-end density
  2. multi_faculty : dept_faculty chunks containing >1 'Designation:' record
  3. tiny          : < 20 tokens of content after the breadcrumb
  4. huge          : > 700 tokens
  5. dup_text      : exact duplicate text across different point IDs
  6. nav_junk      : menu/footer boilerplate (high link-word density)
  7. empty_meta    : missing source_url / document_title

Writes a JSON report to eval/audit_report.json and prints a summary.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from qdrant_client import QdrantClient

COLLECTION = "dtu_rag"
OUT = Path(__file__).parent / "audit_report.json"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SENT_END = re.compile(r"[.!?]\s")
_NAV_WORDS = re.compile(
    r"\b(Home|About Us|Contact Us|Sitemap|Login|Click here|Read More|"
    r"Quick Links|Important Links)\b", re.I)
_DESIGNATION = re.compile(r"Designation\s*:", re.I)


def sentence_end_density(text: str) -> float:
    """Sentence ends per 100 words. Normal prose ~4-8; interleaved columns ~0-1."""
    words = len(text.split())
    if words < 60:
        return 99.0
    return 100.0 * len(_SENT_END.findall(text)) / words


def audit(host="localhost", port=6333):
    client = QdrantClient(host=host, port=port)
    points, offset = [], None
    while True:
        batch, offset = client.scroll(COLLECTION, limit=512, offset=offset,
                                      with_payload=True, with_vectors=False)
        points.extend(batch)
        if offset is None:
            break

    issues = defaultdict(list)
    text_seen: dict[str, str] = {}
    by_doc = Counter()
    by_type = Counter()

    for pt in points:
        p = pt.payload
        text = p.get("text", "")
        cid = p.get("chunk_id", str(pt.id))
        title = p.get("document_title", "?")
        by_doc[title] += 1
        by_type[p.get("document_type", "?")] += 1

        body = text.split("\n\n", 1)[-1]
        toks = p.get("token_count", 0)

        if not p.get("source_url") or not p.get("document_title"):
            issues["empty_meta"].append((cid, title, body[:80]))
        if toks and toks < 20:
            issues["tiny"].append((cid, title, body[:80]))
        if toks > 700:
            issues["huge"].append((cid, title, f"{toks} tokens"))
        if p.get("document_type") == "dept_faculty":
            n = len(_DESIGNATION.findall(body))
            if n > 1:
                issues["multi_faculty"].append((cid, title, f"{n} faculty in one chunk: {body[:100]}"))
        d = sentence_end_density(body)
        if d < 1.2 and p.get("block_type") == "prose" and len(body.split()) > 120:
            issues["interleaved_or_runon"].append((cid, title, f"density={d:.2f} {body[:100]}"))
        nav_hits = len(_NAV_WORDS.findall(body))
        if nav_hits >= 4:
            issues["nav_junk"].append((cid, title, f"{nav_hits} nav words: {body[:100]}"))
        if body in text_seen and text_seen[body] != cid:
            issues["dup_text"].append((cid, title, f"dup of {text_seen[body]}: {body[:80]}"))
        else:
            text_seen[body] = cid

    print(f"Total points: {len(points)}")
    print("\nBy document_type:")
    for k, v in by_type.most_common():
        print(f"  {k:20s} {v}")
    print("\nBy document (top 25):")
    for k, v in by_doc.most_common(25):
        print(f"  {str(k)[:60]:60s} {v}")
    print("\nIssues:")
    for k, v in sorted(issues.items()):
        print(f"  {k:25s} {len(v)}")
        for cid, title, snip in v[:5]:
            print(f"      [{cid}] {str(title)[:40]} | {snip[:90]}")

    OUT.write_text(json.dumps(
        {k: [{"chunk_id": c, "doc": t, "detail": s} for c, t, s in v]
         for k, v in issues.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report: {OUT}")


if __name__ == "__main__":
    audit()
