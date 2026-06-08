"""
Pipeline resumability helpers.
All pipeline scripts import from here instead of writing their own queries.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from manifest.manifest import ManifestDB, VALID_STAGES
from manifest.queries import get_pending, get_running

# ---------------------------------------------------------------------------
# Named SQL for work queues  (stage → SELECT query)
# ---------------------------------------------------------------------------
WORK_QUEUE_SQL: dict[str, str] = {
    stage: (
        f"SELECT * FROM documents "
        f"WHERE {stage}_status = 'pending' "
        f"ORDER BY id"
    )
    for stage in VALID_STAGES
}


def recover_and_get_pending(
    db: ManifestDB,
    stage: str,
) -> list[sqlite3.Row]:
    """
    Crash-recovery + pending fetch in one call.

    1. Reset any 'running' rows back to 'pending' (handles killed/crashed workers).
    2. Return all 'pending' rows for the given stage.

    Call this at the start of every pipeline worker process.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage: {stage!r}")

    # Step 1: crash recovery — running → pending
    db._conn.execute(
        f"UPDATE documents SET {stage}_status = 'pending' "
        f"WHERE {stage}_status = 'running'"
    )
    db._conn.commit()

    # Step 2: return pending work
    return get_pending(db, stage)
