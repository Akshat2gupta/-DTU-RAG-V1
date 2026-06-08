"""
Utility to extract HTTP metadata from a Scrapy response.
"""
from __future__ import annotations

import datetime
from email.utils import parsedate_to_datetime
from typing import Any


def extract_http_meta(response) -> dict[str, Any]:
    """
    Extract HTTP metadata from a Scrapy response object.

    Returns a dict with keys:
        last_modified   - ISO 8601 string or None
        etag            - string or None
        content_type    - string or None
        content_length  - int or None
        crawl_timestamp - ISO 8601 UTC string (always present)
    """
    def _header(name: str) -> str | None:
        raw = response.headers.get(name)
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", errors="replace").strip()
        return str(raw).strip()

    # Last-Modified → ISO 8601
    last_modified: str | None = None
    lm_raw = _header("Last-Modified")
    if lm_raw:
        try:
            dt = parsedate_to_datetime(lm_raw)
            last_modified = dt.astimezone(datetime.timezone.utc).isoformat()
        except Exception:
            last_modified = lm_raw  # store raw string if parsing fails

    # Content-Length → int
    content_length: int | None = None
    cl_raw = _header("Content-Length")
    if cl_raw:
        try:
            content_length = int(cl_raw)
        except ValueError:
            pass

    # Crawl timestamp (UTC now)
    crawl_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "last_modified":   last_modified,
        "etag":            _header("ETag"),
        "content_type":    _header("Content-Type"),
        "content_length":  content_length,
        "crawl_timestamp": crawl_timestamp,
    }
