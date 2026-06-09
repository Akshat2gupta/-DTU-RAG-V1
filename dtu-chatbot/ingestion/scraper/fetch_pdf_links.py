#!/usr/bin/env python3
"""
Targeted PDF-link fetcher for dtu.ac.in pages that block Scrapy/Twisted.

Uses requests (urllib3/OpenSSL) which DTU's server accepts, unlike Twisted.
Fetches the ordinance and notice index pages, extracts all PDF links,
and upserts them into manifest.db.

Run from dtu-chatbot/:
    python ingestion/scraper/fetch_pdf_links.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from manifest.manifest import ManifestDB
from ingestion.scraper.url_rules import is_allowed, is_pdf

MANIFEST_DB_PATH = _PROJECT_ROOT / "manifest" / "manifest.db"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Pages to scrape: (url, category, doc_type)
SEED_PAGES = [
    ("https://dtu.ac.in/Web/Academics/ordinance.php",   "ordinance_html",   "ordinance"),
    ("https://dtu.ac.in/Web/Academics/notice.php",      "notice_html",      "notice"),
    ("https://dtu.ac.in/Web/Academics/scholarship.php", "scholarship_html", "scholarship"),
]


def _fetch(url: str, session: requests.Session) -> str | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  WARN: {url} → {e}")
        return None


def _extract_pdf_links(html: str, base_url: str) -> list[tuple[str, str | None]]:
    """Return list of (absolute_pdf_url, date_label_or_None)."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str | None]] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        url = urljoin(base_url, href)
        if not is_pdf(url) or not is_allowed(url):
            continue
        # Try to find a date label in the surrounding row (notice page pattern)
        date_label = None
        tr = a.find_parent("tr")
        if tr:
            tds = tr.find_all("td")
            if len(tds) >= 2:
                date_label = tds[0].get_text(strip=True) or None
        results.append((url, date_label))

    return results


def main() -> None:
    MANIFEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = ManifestDB(MANIFEST_DB_PATH)
    session = requests.Session()

    total_new = 0
    for page_url, category, doc_type in SEED_PAGES:
        print(f"\nFetching {page_url} ...")
        html = _fetch(page_url, session)
        if not html:
            continue

        # Insert the index page itself
        db.insert_document(
            url=page_url,
            category=category,
            document_type=doc_type,
            scrape_status="done",
        )

        # Extract and insert PDF links
        pdf_links = _extract_pdf_links(html, page_url)
        print(f"  Found {len(pdf_links)} PDF links")
        new_count = 0
        for pdf_url, date_label in pdf_links:
            row_id = db.insert_document(
                url=pdf_url,
                category=category.replace("_html", "_pdf"),
                document_type=doc_type,
                date_published=date_label,
                scrape_status="done",
            )
            if row_id:
                new_count += 1
        print(f"  Inserted {new_count} new rows into manifest.db")
        total_new += new_count

    db.close()
    print(f"\nDone. Total new PDFs added to manifest.db: {total_new}")

    # Print manifest stats
    db2 = ManifestDB(MANIFEST_DB_PATH)
    stats = db2.get_stats()
    db2.close()
    print(f"Manifest totals: {stats['total']} rows, download_pending={stats.get('download', {}).get('pending', 0)}")


if __name__ == "__main__":
    main()
