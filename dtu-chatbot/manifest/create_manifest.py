"""
CLI script: create (or verify) the manifest SQLite database.

Usage:
    python manifest/create_manifest.py
    python manifest/create_manifest.py --db-path /custom/path/manifest.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure dtu-chatbot/ is on sys.path regardless of how this script is invoked
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from manifest.manifest import ManifestDB


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the DTU corpus manifest database."
    )
    parser.add_argument(
        "--db-path",
        default="manifest/manifest.db",
        help="Path to the SQLite database file (default: manifest/manifest.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with ManifestDB(db_path) as db:
        stats = db.get_stats()

    print(f"Manifest database ready at: {db_path.resolve()}")
    print(f"Total documents: {stats.get('total', 0)}")


if __name__ == "__main__":
    main()
