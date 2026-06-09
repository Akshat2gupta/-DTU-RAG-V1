#!/usr/bin/env python3
"""
One-time migration: set batch_year on all existing Qdrant points.

Uses qdrant_client.set_payload (no re-embedding needed).
Scrolls through all points, reads source_url / document_title,
extracts the 4-digit year and patches the payload in batches.

Run once from Rag_V1/:
    python vertical_slice/backfill_batch_year.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_CHATBOT   = _REPO_ROOT / "dtu-chatbot"

for _p in (str(_CHATBOT), str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

from qdrant_client import QdrantClient
from qdrant_client.models import SetPayload

COLLECTION_NAME = "dtu_rag"
_YEAR_RE = re.compile(r"20(1[5-9]|2[0-9])")   # 2015–2029
SCROLL_BATCH = 500


def _extract_year(payload: dict) -> int:
    for key in ("document_title", "source_url", "date_published"):
        text = payload.get(key) or ""
        m = _YEAR_RE.search(text)
        if m:
            return int(m.group(0))
    return 0


def main(host: str = "localhost", port: int = 6333) -> None:
    qdrant = QdrantClient(host=host, port=port)
    info   = qdrant.get_collection(COLLECTION_NAME)
    total  = info.points_count or 0
    print(f"Collection '{COLLECTION_NAME}': {total} points")

    processed = 0
    updated   = 0
    offset    = None

    while True:
        result, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            break

        # Group points by the year value we want to set
        by_year: dict[int, list] = {}
        for pt in result:
            yr = _extract_year(pt.payload or {})
            by_year.setdefault(yr, []).append(pt.id)

        for yr, ids in by_year.items():
            qdrant.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"batch_year": yr},
                points=ids,
            )
            if yr != 0:
                updated += len(ids)

        processed += len(result)
        print(f"  Processed {processed}/{total}  (updated non-zero: {updated})", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\nDone. {processed} points processed, {updated} set to a specific year.")
    # Show year distribution
    dist: dict[int, int] = {}
    for pt_id in range(0, processed, 1):
        pass  # we can't easily get this without a second scroll

    print("\nChecking year distribution (first 2000 points)...")
    result, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=2000,
        with_payload=["batch_year"],
        with_vectors=False,
    )
    yr_dist: dict[int, int] = {}
    for pt in result:
        yr = (pt.payload or {}).get("batch_year", 0)
        yr_dist[yr] = yr_dist.get(yr, 0) + 1
    for yr in sorted(yr_dist):
        label = str(yr) if yr else "0 (evergreen)"
        print(f"  batch_year={label}: {yr_dist[yr]} points")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    args = ap.parse_args()
    main(args.qdrant_host, args.qdrant_port)
