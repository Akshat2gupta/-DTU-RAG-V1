"""
SQLite schema for the DTU corpus manifest database.
"""
from __future__ import annotations

import sqlite3

# ---------------------------------------------------------------------------
# Category → document-type mapping
# ---------------------------------------------------------------------------
CATEGORY_TO_DOCTYPE: dict[str, str] = {
    "ordinance_pdf":     "ordinance",
    "ordinance_html":    "ordinance_index",
    "notice_pdf":        "notice",
    "notice_html":       "notice_index",
    "scholarship_pdf":   "scholarship",
    "scholarship_html":  "scholarship_index",
    "dept_about":        "dept_about",
    "dept_scheme":       "dept_scheme",
    "dept_faculty":      "dept_faculty",
    "hostel_html":       "hostel_info",
    "saarthi_html":      "admission_info",
    "exam_html":         "exam_info",
    "unknown":           "unknown",
}

# ---------------------------------------------------------------------------
# Semantic keyword → sub-type classification for notices
# ---------------------------------------------------------------------------
NOTICE_KEYWORD_TYPES: dict[str, str] = {
    "admission":     "admission_notice",
    "scholarship":   "scholarship_notice",
    "fee":           "fee_notice",
    "examination":   "exam_notice",
    "exam":          "exam_notice",
    "datesheet":     "datesheet",
    "result":        "result_notice",
    "placement":     "placement_notice",
    "convocation":   "convocation_notice",
    "holiday":       "holiday_notice",
    "circular":      "circular",
    "tender":        "tender",
    "recruitment":   "recruitment_notice",
}

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    url              TEXT    NOT NULL UNIQUE,
    file_hash        TEXT,
    category         TEXT,
    document_type    TEXT,
    title            TEXT,
    date_published   TEXT,
    date_scraped     TEXT,
    language         TEXT    DEFAULT 'en',
    is_ocr           INTEGER DEFAULT 0,
    is_duplicate     INTEGER DEFAULT 0,

    -- Pipeline stage columns: pending | running | done | failed | skipped
    scrape_status    TEXT    DEFAULT 'pending',
    scrape_notes     TEXT,

    download_status  TEXT    DEFAULT 'pending',
    download_notes   TEXT,

    parse_status     TEXT    DEFAULT 'pending',
    parse_notes      TEXT,

    clean_status     TEXT    DEFAULT 'pending',
    clean_notes      TEXT,

    chunk_status     TEXT    DEFAULT 'pending',
    chunk_notes      TEXT,

    embed_status     TEXT    DEFAULT 'pending',
    embed_notes      TEXT,

    index_status     TEXT    DEFAULT 'pending',
    index_notes      TEXT,

    file_path        TEXT,
    file_size        INTEGER,
    page_count       INTEGER,

    created_at       TEXT    DEFAULT (datetime('now')),
    updated_at       TEXT    DEFAULT (datetime('now'))
);
"""

CREATE_UPDATED_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_documents_updated_at
AFTER UPDATE ON documents
FOR EACH ROW
BEGIN
    UPDATE documents SET updated_at = datetime('now') WHERE id = OLD.id;
END;
"""

CREATE_URL_INDEX     = "CREATE INDEX IF NOT EXISTS idx_documents_url      ON documents(url);"
CREATE_HASH_INDEX    = "CREATE INDEX IF NOT EXISTS idx_documents_hash     ON documents(file_hash);"
CREATE_STAGE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_scrape_status   ON documents(scrape_status);",
    "CREATE INDEX IF NOT EXISTS idx_download_status ON documents(download_status);",
    "CREATE INDEX IF NOT EXISTS idx_parse_status    ON documents(parse_status);",
    "CREATE INDEX IF NOT EXISTS idx_clean_status    ON documents(clean_status);",
    "CREATE INDEX IF NOT EXISTS idx_chunk_status    ON documents(chunk_status);",
    "CREATE INDEX IF NOT EXISTS idx_embed_status    ON documents(embed_status);",
    "CREATE INDEX IF NOT EXISTS idx_index_status    ON documents(index_status);",
]


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, triggers, and indexes."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(CREATE_DOCUMENTS_TABLE)
    conn.execute(CREATE_UPDATED_TRIGGER)
    conn.execute(CREATE_URL_INDEX)
    conn.execute(CREATE_HASH_INDEX)
    for stmt in CREATE_STAGE_INDEXES:
        conn.execute(stmt)
    conn.commit()


# ---------------------------------------------------------------------------
# Default stage values per document origin
# ---------------------------------------------------------------------------
PDF_DEFAULTS = {
    "download_status": "pending",
    "parse_status":    "pending",
    "clean_status":    "skipped",   # PDFs go through PDF-specific clean, not HTML clean
}

HTML_DEFAULTS = {
    "download_status": "skipped",   # HTML body already captured by spider
    "parse_status":    "skipped",   # Parsing done inline during scrape
}
