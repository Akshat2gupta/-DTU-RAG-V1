"""
ManifestDB: thin wrapper around the SQLite corpus manifest.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from manifest.schema import (
    HTML_DEFAULTS,
    PDF_DEFAULTS,
    create_schema,
)

VALID_STAGES = (
    "scrape", "download", "parse", "clean", "chunk", "embed", "index"
)
VALID_STATUSES = ("pending", "running", "done", "failed", "skipped")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ManifestDB:
    """Thread-unsafe single-connection manifest database."""

    def __init__(self, db_path: str | Path = "manifest.db"):
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        create_schema(self._conn)

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_document(
        self,
        url: str,
        category: str,
        document_type: str,
        *,
        title: str | None = None,
        date_published: str | None = None,
        date_scraped: str | None = None,
        language: str = "en",
        file_path: str | None = None,
        file_size: int | None = None,
        scrape_status: str = "pending",
        scrape_notes: str | None = None,
    ) -> int | None:
        """
        INSERT OR IGNORE on url (safe for re-crawl).
        Returns the rowid of the inserted row, or None if url already existed.
        """
        is_pdf = category.endswith("_pdf")
        defaults = PDF_DEFAULTS if is_pdf else HTML_DEFAULTS

        row = {
            "url":            url,
            "category":       category,
            "document_type":  document_type,
            "title":          title,
            "date_published": date_published,
            "date_scraped":   date_scraped or _now_iso(),
            "language":       language,
            "file_path":      file_path,
            "file_size":      file_size,
            "scrape_status":  scrape_status,
            "scrape_notes":   scrape_notes,
            **defaults,
        }

        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT OR IGNORE INTO documents ({cols}) VALUES ({placeholders})"

        cur = self._conn.execute(sql, list(row.values()))
        self._conn.commit()
        return cur.lastrowid if cur.rowcount else None

    # ------------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------------

    def update_stage(
        self,
        url: str,
        stage: str,
        status: str,
        notes: str | None = None,
    ) -> None:
        """Set <stage>_status (and optional notes) for a document by url."""
        if stage not in VALID_STAGES:
            raise ValueError(f"Unknown stage: {stage!r}")
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown status: {status!r}")

        self._conn.execute(
            f"UPDATE documents SET {stage}_status=?, {stage}_notes=? WHERE url=?",
            (status, notes, url),
        )
        self._conn.commit()

    def mark_running(self, url: str, stage: str) -> None:
        self.update_stage(url, stage, "running")

    def mark_duplicate(self, url: str, canonical_url: str | None = None) -> None:
        """Mark a document as duplicate; cascade all remaining stages to 'skipped'."""
        self._conn.execute(
            """UPDATE documents SET
                is_duplicate     = 1,
                download_status  = CASE WHEN download_status = 'pending' THEN 'skipped' ELSE download_status END,
                parse_status     = CASE WHEN parse_status    = 'pending' THEN 'skipped' ELSE parse_status    END,
                clean_status     = CASE WHEN clean_status    = 'pending' THEN 'skipped' ELSE clean_status    END,
                chunk_status     = CASE WHEN chunk_status    = 'pending' THEN 'skipped' ELSE chunk_status    END,
                embed_status     = CASE WHEN embed_status    = 'pending' THEN 'skipped' ELSE embed_status    END,
                index_status     = CASE WHEN index_status    = 'pending' THEN 'skipped' ELSE index_status    END,
                scrape_notes     = COALESCE(scrape_notes, ?)
            WHERE url = ?""",
            (f"duplicate_of:{canonical_url}" if canonical_url else "duplicate", url),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_by_url(self, url: str) -> sqlite3.Row | None:
        cur = self._conn.execute(
            "SELECT * FROM documents WHERE url = ?", (url,)
        )
        return cur.fetchone()

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate counts per stage/status."""
        stats: dict[str, Any] = {}
        for stage in VALID_STAGES:
            cur = self._conn.execute(
                f"SELECT {stage}_status, COUNT(*) as cnt FROM documents "
                f"GROUP BY {stage}_status"
            )
            stats[stage] = {row[0]: row[1] for row in cur.fetchall()}
        cur = self._conn.execute("SELECT COUNT(*) FROM documents")
        stats["total"] = cur.fetchone()[0]
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE is_duplicate = 1"
        )
        stats["duplicates"] = cur.fetchone()[0]
        return stats

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
