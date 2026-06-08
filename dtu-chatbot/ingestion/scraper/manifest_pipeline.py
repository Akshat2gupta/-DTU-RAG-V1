"""
Scrapy pipeline: writes CrawlRecord items to a JSONL manifest and saves
HTML bodies to data/raw/html/.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ingestion.scraper.items import CrawlRecord

# Project root is three levels up from this file
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent  # ingestion/scraper → ingestion → dtu-chatbot

HTML_DIR  = _PROJECT_ROOT / "data" / "raw" / "html"
LOGS_DIR  = _PROJECT_ROOT / "logs"


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


class ManifestPipeline:
    """
    Appends one JSONL record per CrawlRecord.
    HTML bodies are saved separately; only the file path goes into JSONL.
    """

    def open_spider(self, spider):
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        manifest_path = LOGS_DIR / f"crawl_manifest_{ts}.jsonl"
        self._fh = open(manifest_path, "a", encoding="utf-8", buffering=1)
        spider.logger.info(f"ManifestPipeline writing to {manifest_path}")

    def close_spider(self, spider):
        self._fh.close()

    def process_item(self, item: CrawlRecord, spider):
        record: dict = dict(item)

        # Save HTML body separately
        html_body = record.pop("html_body", None)
        if html_body:
            h = _url_hash(record["url"])
            html_path = HTML_DIR / f"{h}.html"
            html_path.write_text(html_body, encoding="utf-8")
            record["html_body_path"] = str(html_path.relative_to(_PROJECT_ROOT))
        else:
            record.setdefault("html_body_path", None)

        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return item
