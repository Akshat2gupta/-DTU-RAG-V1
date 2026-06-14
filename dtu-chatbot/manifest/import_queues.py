
"""
Import crawler JSONL manifest output into the SQLite manifest database.

Usage:
    python manifest/import_queues.py logs/crawl_manifest_20240101T120000.jsonl
    python manifest/import_queues.py logs/*.jsonl --db-path manifest/manifest.db
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from manifest.manifest import ManifestDB
from manifest.schema import CATEGORY_TO_DOCTYPE


def import_jsonl(db: ManifestDB, jsonl_path: Path) -> tuple[int, int]:
    """
    Read a JSONL crawl manifest and insert records into the DB.
    Returns (inserted, skipped) counts.
    """
    inserted = 0
    skipped = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            url = record.get("url")
            if not url:
                skipped += 1
                continue

            category     = record.get("category", "unknown")
            document_type = CATEGORY_TO_DOCTYPE.get(category, "unknown")

            result = db.insert_document(
                url=url,
                category=category,
                document_type=document_type,
                date_scraped=record.get("crawl_timestamp"),
                scrape_status="done",
                scrape_notes=None,
            )
            if result is not None:
                inserted += 1
            else:
                skipped += 1

    return inserted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import crawler JSONL manifest into SQLite."
    )
    parser.add_argument(
        "jsonl_files",
        nargs="+",
        help="One or more JSONL crawl manifest files to import.",
    )
    parser.add_argument(
        "--db-path",
        default="manifest/manifest.db",
        help="Path to the SQLite database (default: manifest/manifest.db)",
    )
    args = parser.parse_args()

    total_inserted = 0
    total_skipped  = 0

    with ManifestDB(args.db_path) as db:
        for pattern in args.jsonl_files:
            for path in sorted(Path(".").glob(pattern)):
                ins, skp = import_jsonl(db, path)
                print(f"  {path}: {ins} inserted, {skp} skipped/duplicate")
                total_inserted += ins
                total_skipped  += skp

    print(f"\nImport complete: {total_inserted} inserted, {total_skipped} skipped/duplicate")


if __name__ == "__main__":
    main()
