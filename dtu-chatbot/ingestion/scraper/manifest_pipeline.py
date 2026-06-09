"""
Scrapy pipeline: writes CrawlRecord items to manifest.db (SQLite) and saves
HTML bodies to data/raw/html/. Also appends a JSONL audit log per crawl run.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ingestion.scraper.items import CrawlRecord
from manifest.manifest import ManifestDB

# Project root is three levels up from this file
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent  # ingestion/scraper -> ingestion -> dtu-chatbot

HTML_DIR  = _PROJECT_ROOT / "data" / "raw" / "html"
LOGS_DIR  = _PROJECT_ROOT / "logs"
MANIFEST_DB_PATH = _PROJECT_ROOT / "manifest" / "manifest.db"


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


class ManifestPipeline:
    """
    Primary output: inserts/updates rows in manifest.db via ManifestDB.
    Secondary output: appends JSONL audit log for each crawl run.
    HTML bodies are saved to data/raw/html/<url_hash>.html.
    """

    def open_spider(self, spider):
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        self._db = ManifestDB(MANIFEST_DB_PATH)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        audit_path = LOGS_DIR / f"crawl_manifest_{ts}.jsonl"
        self._fh = open(audit_path, "a", encoding="utf-8", buffering=1)
        spider.logger.info(f"ManifestPipeline: db={MANIFEST_DB_PATH}  audit={audit_path}")

    def close_spider(self, spider):
        self._db.close()
        self._fh.close()

    def process_item(self, item: CrawlRecord, spider):
        record: dict = dict(item)

        # -- Save HTML body -------------------------------------------------
        html_body = record.pop("html_body", None)
        html_body_path = None
        if html_body:
            h = _url_hash(record["url"])
            html_path = HTML_DIR / f"{h}.html"
            html_path.write_text(html_body, encoding="utf-8")
            html_body_path = str(html_path.relative_to(_PROJECT_ROOT))
            record["html_body_path"] = html_body_path
        else:
            record.setdefault("html_body_path", None)

        # -- Write audit JSONL ----------------------------------------------
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # -- Write to manifest.db -------------------------------------------
        url      = record["url"]
        category = record.get("category", "unknown")
        doc_type = record.get("doc_type", "unknown")
        status   = record.get("http_status", 0)

        # Use link_date_label if present (notice dates), else last_modified
        date_published = record.get("link_date_label") or record.get("last_modified")

        # file_path: html_body_path for HTML pages, None for PDF stubs
        # (PDF file_path gets set later by download_worker)
        file_path = html_body_path

        scrape_status = "done" if status == 200 else "failed"
        scrape_notes  = json.dumps({
            "http_status":   status,
            "content_type":  record.get("content_type"),
            "etag":          record.get("etag"),
            "content_length": record.get("content_length"),
        })

        self._db.insert_document(
            url=url,
            category=category,
            document_type=doc_type,
            date_published=date_published,
            date_scraped=record.get("crawl_timestamp"),
            file_path=file_path,
            scrape_status=scrape_status,
            scrape_notes=scrape_notes,
        )

        return item
