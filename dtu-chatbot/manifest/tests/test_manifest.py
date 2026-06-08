"""
Tests for schema.py, manifest.py, queries.py, resumability.py.
Uses in-memory SQLite — no file I/O required.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sqlite3
import pytest

from manifest.manifest import ManifestDB, VALID_STAGES
from manifest.queries import (
    get_pending,
    get_running,
    get_failed,
    get_dashboard_stats,
    get_by_document_type,
    get_recent,
)
from manifest.resumability import recover_and_get_pending, WORK_QUEUE_SQL
from manifest.schema import CATEGORY_TO_DOCTYPE, PDF_DEFAULTS, HTML_DEFAULTS


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def db():
    """In-memory ManifestDB — isolated per test."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    m = ManifestDB.__new__(ManifestDB)
    m._path = ":memory:"
    m._conn = conn

    from manifest.schema import create_schema
    conn.execute("PRAGMA journal_mode=WAL;")
    create_schema(conn)
    yield m
    conn.close()


def _insert(db: ManifestDB, url: str, category: str = "ordinance_pdf",
            doc_type: str = "ordinance") -> int | None:
    return db.insert_document(url=url, category=category, document_type=doc_type)


# ===========================================================================
# Schema / INSERT tests
# ===========================================================================

class TestInsert:
    def test_basic_insert_returns_rowid(self, db):
        rowid = _insert(db, "https://dtu.ac.in/test1.pdf")
        assert rowid is not None
        assert rowid > 0

    def test_insert_or_ignore_deduplication(self, db):
        url = "https://dtu.ac.in/test2.pdf"
        r1 = _insert(db, url)
        r2 = _insert(db, url)
        assert r1 is not None
        assert r2 is None  # duplicate — ignored

    def test_duplicate_url_not_doubled(self, db):
        url = "https://dtu.ac.in/test3.pdf"
        _insert(db, url)
        _insert(db, url)
        cur = db._conn.execute("SELECT COUNT(*) FROM documents WHERE url=?", (url,))
        assert cur.fetchone()[0] == 1

    def test_pdf_defaults_applied(self, db):
        url = "https://dtu.ac.in/ordinance/ug.pdf"
        _insert(db, url, category="ordinance_pdf")
        row = db.get_by_url(url)
        assert row["download_status"] == "pending"
        assert row["parse_status"]    == "pending"
        assert row["clean_status"]    == "skipped"

    def test_html_defaults_applied(self, db):
        url = "https://dtu.ac.in/Web/Academics/ordinance.php"
        _insert(db, url, category="ordinance_html", doc_type="ordinance_index")
        row = db.get_by_url(url)
        assert row["download_status"] == "skipped"
        assert row["parse_status"]    == "skipped"

    def test_multiple_inserts_unique_rowids(self, db):
        ids = [_insert(db, f"https://dtu.ac.in/doc{i}.pdf") for i in range(5)]
        assert len(set(ids)) == 5

    def test_scrape_status_defaults_to_pending(self, db):
        _insert(db, "https://dtu.ac.in/a.pdf")
        row = db.get_by_url("https://dtu.ac.in/a.pdf")
        assert row["scrape_status"] == "pending"


# ===========================================================================
# Stage transition tests
# ===========================================================================

class TestStageTransitions:
    def test_update_stage_pending_to_running(self, db):
        url = "https://dtu.ac.in/t1.pdf"
        _insert(db, url)
        db.update_stage(url, "download", "running")
        row = db.get_by_url(url)
        assert row["download_status"] == "running"

    def test_update_stage_running_to_done(self, db):
        url = "https://dtu.ac.in/t2.pdf"
        _insert(db, url)
        db.update_stage(url, "download", "running")
        db.update_stage(url, "download", "done")
        row = db.get_by_url(url)
        assert row["download_status"] == "done"

    def test_update_stage_with_notes(self, db):
        url = "https://dtu.ac.in/t3.pdf"
        _insert(db, url)
        db.update_stage(url, "parse", "failed", notes="timeout error")
        row = db.get_by_url(url)
        assert row["parse_status"] == "failed"
        assert "timeout" in row["parse_notes"]

    def test_mark_running_helper(self, db):
        url = "https://dtu.ac.in/t4.pdf"
        _insert(db, url)
        db.mark_running(url, "embed")
        row = db.get_by_url(url)
        assert row["embed_status"] == "running"

    def test_invalid_stage_raises(self, db):
        url = "https://dtu.ac.in/t5.pdf"
        _insert(db, url)
        with pytest.raises(ValueError):
            db.update_stage(url, "nonexistent_stage", "done")

    def test_invalid_status_raises(self, db):
        url = "https://dtu.ac.in/t6.pdf"
        _insert(db, url)
        with pytest.raises(ValueError):
            db.update_stage(url, "download", "in_progress")

    def test_all_valid_stages_cycle(self, db):
        url = "https://dtu.ac.in/cycle.pdf"
        _insert(db, url)
        for stage in VALID_STAGES:
            db.update_stage(url, stage, "done")
        row = db.get_by_url(url)
        for stage in VALID_STAGES:
            assert row[f"{stage}_status"] == "done"


# ===========================================================================
# Crash recovery tests
# ===========================================================================

class TestCrashRecovery:
    def test_running_reset_to_pending_on_recovery(self, db):
        for i in range(3):
            url = f"https://dtu.ac.in/crash{i}.pdf"
            _insert(db, url)
            db.update_stage(url, "download", "running")

        pending = recover_and_get_pending(db, "download")
        # All 3 "running" rows should now be pending
        assert len(pending) == 3

    def test_done_rows_not_reset(self, db):
        url_done    = "https://dtu.ac.in/done.pdf"
        url_running = "https://dtu.ac.in/running.pdf"
        _insert(db, url_done)
        _insert(db, url_running)
        db.update_stage(url_done,    "parse", "done")
        db.update_stage(url_running, "parse", "running")

        pending = recover_and_get_pending(db, "parse")
        urls = [r["url"] for r in pending]
        assert url_running in urls
        assert url_done    not in urls

    def test_recovery_idempotent(self, db):
        url = "https://dtu.ac.in/idempotent.pdf"
        _insert(db, url)
        db.update_stage(url, "chunk", "running")

        # Call twice — should not error or double-reset
        recover_and_get_pending(db, "chunk")
        pending = recover_and_get_pending(db, "chunk")
        assert len(pending) == 1


# ===========================================================================
# mark_duplicate tests
# ===========================================================================

class TestMarkDuplicate:
    def test_duplicate_flag_set(self, db):
        url = "https://dtu.ac.in/dup.pdf"
        _insert(db, url)
        db.mark_duplicate(url)
        row = db.get_by_url(url)
        assert row["is_duplicate"] == 1

    def test_duplicate_cascades_pending_to_skipped(self, db):
        url = "https://dtu.ac.in/dup2.pdf"
        _insert(db, url)
        db.mark_duplicate(url)
        row = db.get_by_url(url)
        for stage in ("download", "parse", "clean", "chunk", "embed", "index"):
            status = row[f"{stage}_status"]
            assert status in ("skipped", "pending", "done"), (
                f"{stage}_status={status!r} should be skipped for duplicate"
            )

    def test_duplicate_with_canonical_url(self, db):
        url       = "https://dtu.ac.in/dup3.pdf"
        canonical = "https://dtu.ac.in/original.pdf"
        _insert(db, url)
        db.mark_duplicate(url, canonical_url=canonical)
        row = db.get_by_url(url)
        assert row["is_duplicate"] == 1
        assert canonical in (row["scrape_notes"] or "")


# ===========================================================================
# Query function tests
# ===========================================================================

class TestQueries:
    def test_get_pending_returns_pending_rows(self, db):
        for i in range(3):
            _insert(db, f"https://dtu.ac.in/pend{i}.pdf")
        results = get_pending(db, "download")
        assert len(results) >= 3

    def test_get_running_returns_running_rows(self, db):
        url = "https://dtu.ac.in/run.pdf"
        _insert(db, url)
        db.update_stage(url, "embed", "running")
        results = get_running(db, "embed")
        urls = [r["url"] for r in results]
        assert url in urls

    def test_get_failed_returns_failed_rows(self, db):
        url = "https://dtu.ac.in/fail.pdf"
        _insert(db, url)
        db.update_stage(url, "chunk", "failed", notes="oom")
        results = get_failed(db, "chunk")
        assert any(r["url"] == url for r in results)

    def test_get_dashboard_stats_total(self, db):
        for i in range(4):
            _insert(db, f"https://dtu.ac.in/stat{i}.pdf")
        stats = get_dashboard_stats(db)
        assert stats["total"] >= 4

    def test_get_dashboard_stats_by_category(self, db):
        _insert(db, "https://dtu.ac.in/cat1.pdf", category="notice_pdf")
        _insert(db, "https://dtu.ac.in/cat2.pdf", category="notice_pdf")
        _insert(db, "https://dtu.ac.in/cat3.pdf", category="ordinance_pdf")
        stats = get_dashboard_stats(db)
        assert stats["by_category"].get("notice_pdf", 0) >= 2

    def test_get_by_document_type(self, db):
        _insert(db, "https://dtu.ac.in/dt1.pdf", doc_type="notice")
        _insert(db, "https://dtu.ac.in/dt2.pdf", doc_type="notice")
        _insert(db, "https://dtu.ac.in/dt3.pdf", doc_type="ordinance")
        notices = get_by_document_type(db, "notice")
        assert len(notices) >= 2
        assert all(r["document_type"] == "notice" for r in notices)

    def test_get_recent_returns_rows(self, db):
        _insert(db, "https://dtu.ac.in/recent.pdf")
        results = get_recent(db, days=7)
        assert len(results) >= 1

    def test_get_pending_invalid_stage_raises(self, db):
        with pytest.raises(ValueError):
            get_pending(db, "bad_stage")


# ===========================================================================
# WORK_QUEUE_SQL coverage
# ===========================================================================

class TestWorkQueueSql:
    def test_all_stages_have_sql(self):
        for stage in VALID_STAGES:
            assert stage in WORK_QUEUE_SQL, f"Missing SQL for stage: {stage}"

    def test_sql_executes_without_error(self, db):
        for stage, sql in WORK_QUEUE_SQL.items():
            try:
                db._conn.execute(sql).fetchall()
            except sqlite3.Error as e:
                pytest.fail(f"SQL for stage {stage!r} failed: {e}")


# ===========================================================================
# Schema constants tests
# ===========================================================================

class TestSchemaConstants:
    def test_category_to_doctype_has_pdf_types(self):
        assert "ordinance_pdf"   in CATEGORY_TO_DOCTYPE
        assert "notice_pdf"      in CATEGORY_TO_DOCTYPE
        assert "scholarship_pdf" in CATEGORY_TO_DOCTYPE

    def test_pdf_defaults_has_clean_skipped(self):
        assert PDF_DEFAULTS["clean_status"] == "skipped"

    def test_html_defaults_has_download_skipped(self):
        assert HTML_DEFAULTS["download_status"] == "skipped"
        assert HTML_DEFAULTS["parse_status"]    == "skipped"
