"""
Named query functions for the manifest database.
All SQL lives here — no inline SQL in pipeline scripts.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from manifest.manifest import ManifestDB, VALID_STAGES


def get_pending(db: ManifestDB, stage: str) -> list[sqlite3.Row]:
    """Return all documents where <stage>_status = 'pending'."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage: {stage!r}")
    cur = db._conn.execute(
        f"SELECT * FROM documents WHERE {stage}_status = 'pending' ORDER BY id"
    )
    return cur.fetchall()


def get_running(db: ManifestDB, stage: str) -> list[sqlite3.Row]:
    """Return all documents where <stage>_status = 'running'."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage: {stage!r}")
    cur = db._conn.execute(
        f"SELECT * FROM documents WHERE {stage}_status = 'running' ORDER BY id"
    )
    return cur.fetchall()


def get_failed(db: ManifestDB, stage: str) -> list[sqlite3.Row]:
    """Return all documents where <stage>_status = 'failed'."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage: {stage!r}")
    cur = db._conn.execute(
        f"SELECT * FROM documents WHERE {stage}_status = 'failed' ORDER BY id"
    )
    return cur.fetchall()


def get_dashboard_stats(db: ManifestDB) -> dict[str, Any]:
    """Return dashboard-ready aggregate statistics."""
    stats = db.get_stats()
    # Enrich with per-category counts
    cur = db._conn.execute(
        "SELECT category, COUNT(*) as cnt FROM documents GROUP BY category"
    )
    stats["by_category"] = {row[0]: row[1] for row in cur.fetchall()}
    return stats


def get_by_document_type(db: ManifestDB, doc_type: str) -> list[sqlite3.Row]:
    """Return all documents of a given document_type."""
    cur = db._conn.execute(
        "SELECT * FROM documents WHERE document_type = ? ORDER BY id",
        (doc_type,),
    )
    return cur.fetchall()


def get_ready_to_embed(db: ManifestDB) -> list[sqlite3.Row]:
    """Return documents where chunk_status='done' AND embed_status='pending'."""
    cur = db._conn.execute(
        "SELECT * FROM documents "
        "WHERE chunk_status = 'done' AND embed_status = 'pending' "
        "ORDER BY id"
    )
    return cur.fetchall()


def get_recent(db: ManifestDB, days: int = 7) -> list[sqlite3.Row]:
    """Return documents scraped within the last *days* days."""
    cur = db._conn.execute(
        "SELECT * FROM documents "
        "WHERE date_scraped >= datetime('now', ? || ' days') "
        "ORDER BY date_scraped DESC",
        (f"-{days}",),
    )
    return cur.fetchall()
