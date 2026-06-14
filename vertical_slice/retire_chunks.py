#!/usr/bin/env python3
"""
CLI for chunk retirement (logic lives in ingestion/chunk_retirement.py).

Retirement runs automatically at the end of batch_index.py and ocr_worker.py,
so this CLI is only needed for manual checks:

    python vertical_slice/retire_chunks.py --dry-run
    python vertical_slice/retire_chunks.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_CHATBOT = _HERE.parent / "dtu-chatbot"
for _p in (str(_CHATBOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient

from ingestion.chunk_retirement import COLLECTION, find_retired, retire_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    args = ap.parse_args()

    q = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    before = q.get_collection(COLLECTION).points_count

    rule1, rule2, rule3, rule4 = find_retired(q)
    print(f"Collection points          : {before}")
    print(f"RULE 1 superseded regs     : {len(rule1)}")
    print(f"RULE 2 duplicated sections : {len(rule2)}")
    print(f"RULE 3 heading floods      : {len(rule3)}")
    print(f"RULE 4 ordinance prose     : {len(rule4)}")

    for label, plist in (("RULE 1", rule1), ("RULE 2", rule2),
                         ("RULE 3", rule3), ("RULE 4", rule4)):
        sample = sorted({(p.payload["section_heading"][:55],
                          (p.payload.get("source_url") or "")[-28:]) for p in plist})[:8]
        if sample:
            print(f"\n{label} sample:")
            for h, u in sample:
                print(f"  {h:55s} {u}")

    if args.dry_run:
        n = len({p.id for p in rule1} | {p.id for p in rule2}
                | {p.id for p in rule3} | {p.id for p in rule4})
        print(f"\nDRY RUN — would delete {n} chunks.")
        return

    retire_chunks(q)


if __name__ == "__main__":
    main()
