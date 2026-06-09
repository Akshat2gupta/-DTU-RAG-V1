"""
Stage 3 — PDF download worker.

Reads every document with download_status='pending' from the manifest,
downloads the PDF body, saves it to disk, and updates the manifest atomically.

Duplicate detection is content-based: after download, the SHA-256 is checked
against already-confirmed files; duplicates are marked and the local file removed.

Usage
-----
    cd dtu-chatbot/
    python ingestion/downloader.py
    python ingestion/downloader.py --concurrency 8 --timeout 45
    python ingestion/downloader.py --manifest-path path/to/manifest.db \\
                                   --output-dir data/raw/pdfs

Flow per document
-----------------
    mark_running → HTTP GET (with retries) → save to disk → sha256
    → duplicate check → finalize_download (atomic) or mark_failed
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import httpx

# ── path bootstrap so this runs from dtu-chatbot/ ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from manifest.manifest import ManifestDB
from manifest.resumability import recover_and_get_pending

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STAGE: Final[str] = "download"
PDF_CONTENT_TYPES: Final[frozenset[str]] = frozenset({
    "application/pdf",
    "application/x-pdf",
    "binary/octet-stream",   # some DTU servers mis-label PDFs
})
MIN_PDF_BYTES: Final[int] = 512          # anything smaller is a stub / error page
_SAFE_FILENAME_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\-.]")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    """All runtime tunables in one place — no magic numbers in the worker."""
    manifest_path: Path = Path("manifest.db")
    output_dir: Path    = Path("data/raw/pdfs")
    concurrency: int    = 4
    timeout: float      = 30.0      # seconds per request
    max_retries: int    = 3
    retry_backoff: float = 2.0      # seconds; doubles each attempt
    chunk_size: int     = 65_536    # bytes per read chunk


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class DownloadError(Exception):
    """Base class for all download-worker errors."""


class BadContentTypeError(DownloadError):
    """Server returned a non-PDF Content-Type."""


class EmptyBodyError(DownloadError):
    """Response body is absent or below the minimum size threshold."""


class HttpError(DownloadError):
    """Non-retryable HTTP error (4xx)."""


# ─────────────────────────────────────────────────────────────────────────────
# Filename helpers
# ─────────────────────────────────────────────────────────────────────────────

def _url_to_stem(url: str) -> str:
    """Return a filesystem-safe filename stem derived from the URL path."""
    path = urlparse(url).path.rstrip("/")
    name = Path(path).name or "document"
    # strip extension — we always add .pdf ourselves
    stem = Path(name).stem
    safe = _SAFE_FILENAME_RE.sub("_", stem).strip("_") or "document"
    return safe[:120]   # cap length


def _unique_path(directory: Path, stem: str) -> Path:
    """Return a path that does not yet exist; appends _2, _3 … on collision."""
    candidate = directory / f"{stem}.pdf"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}.pdf"
        if not candidate.exists():
            return candidate
        counter += 1


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 helper
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP download (with retry)
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_pdf(client: httpx.AsyncClient, url: str, cfg: Config) -> bytes:
    """
    Download *url* and return raw bytes.

    Retries on transient network errors and HTTP 5xx responses.
    Raises DownloadError subclasses for non-retryable failures.
    """
    last_exc: Exception | None = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = await client.get(url, follow_redirects=True)

            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            if resp.status_code >= 400:
                raise HttpError(f"HTTP {resp.status_code} for {url}")

            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and content_type not in PDF_CONTENT_TYPES:
                raise BadContentTypeError(
                    f"Expected PDF, got '{content_type}' for {url}"
                )

            body = resp.content
            if len(body) < MIN_PDF_BYTES:
                raise EmptyBodyError(
                    f"Body too small ({len(body)} bytes) for {url}"
                )

            return body

        except (HttpError, BadContentTypeError, EmptyBodyError):
            raise   # never retry client or content errors

        except Exception as exc:
            last_exc = exc
            if attempt < cfg.max_retries:
                wait = cfg.retry_backoff * (2 ** (attempt - 1))
                log.warning(
                    "Download attempt %d/%d failed; retrying in %.1fs",
                    attempt, cfg.max_retries, wait,
                    extra={"url": url, "error": str(exc)},
                )
                await asyncio.sleep(wait)

    raise DownloadError(f"All {cfg.max_retries} attempts failed for {url}") from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Per-document worker
# ─────────────────────────────────────────────────────────────────────────────

async def _download_one(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    db: ManifestDB,
    doc: object,       # sqlite3.Row
    cfg: Config,
    shutdown: asyncio.Event,
) -> None:
    """Download one PDF, update the manifest, log the outcome."""
    url: str = doc["url"]   # type: ignore[index]
    t0 = time.monotonic()

    async with semaphore:
        if shutdown.is_set():
            log.info("Shutdown requested — skipping %s", url, extra={"url": url})
            return

        db.mark_running(url, STAGE)
        log.debug("Starting download", extra={"url": url})

        try:
            body = await _fetch_pdf(client, url, cfg)
        except DownloadError as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            db.update_stage(url, STAGE, "failed", notes=str(exc))
            log.error(
                "FAILED  %s | %s",
                url, exc,
                extra={"url": url, "outcome": "failed", "elapsed_ms": elapsed},
            )
            return
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            db.update_stage(url, STAGE, "failed", notes=f"unexpected: {exc}")
            log.exception(
                "FAILED (unexpected)  %s",
                url,
                extra={"url": url, "outcome": "failed", "elapsed_ms": elapsed},
            )
            return

        # ── content-based duplicate check ────────────────────────────────────
        file_hash = _sha256_of_bytes(body)
        canonical = db.get_document_by_hash(file_hash, exclude_url=url)
        if canonical is not None:
            elapsed = int((time.monotonic() - t0) * 1000)
            db.mark_duplicate(url, canonical_url=canonical["url"])
            log.info(
                "DUPLICATE  %s | canonical=%s",
                url, canonical["url"],
                extra={
                    "url": url,
                    "outcome": "duplicate",
                    "canonical": canonical["url"],
                    "elapsed_ms": elapsed,
                },
            )
            return

        # ── save to disk ─────────────────────────────────────────────────────
        stem = _url_to_stem(url)
        dest = _unique_path(cfg.output_dir, stem)

        try:
            dest.write_bytes(body)
        except OSError as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            db.update_stage(url, STAGE, "failed", notes=f"disk error: {exc}")
            log.error(
                "FAILED (disk)  %s | %s",
                url, exc,
                extra={"url": url, "outcome": "failed_disk", "elapsed_ms": elapsed},
            )
            return

        # ── finalize manifest (atomic) ────────────────────────────────────────
        size = len(body)
        try:
            db.finalize_download(url, STAGE, dest, size, file_hash)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            elapsed = int((time.monotonic() - t0) * 1000)
            db.update_stage(url, STAGE, "failed", notes=f"manifest error: {exc}")
            log.error(
                "FAILED (manifest)  %s | %s",
                url, exc,
                extra={"url": url, "outcome": "failed_manifest", "elapsed_ms": elapsed},
            )
            return

        elapsed = int((time.monotonic() - t0) * 1000)
        log.info(
            "OK  %s | %d KB | %dms",
            url, size // 1024, elapsed,
            extra={
                "url": url,
                "outcome": "ok",
                "bytes_written": size,
                "elapsed_ms": elapsed,
                "path": str(dest),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main async entrypoint
# ─────────────────────────────────────────────────────────────────────────────

async def run(cfg: Config) -> None:
    """Full download pass: recover → fetch pending → download all → summarise."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    shutdown = asyncio.Event()

    def _handle_signal(sig: int, _frame: object) -> None:
        log.warning(
            "Signal %s received — finishing in-flight downloads then exiting",
            signal.Signals(sig).name,
        )
        shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    with ManifestDB(cfg.manifest_path) as db:
        pending = recover_and_get_pending(db, STAGE)
        total = len(pending)

        if total == 0:
            log.info("No pending downloads — nothing to do.")
            return

        log.info(
            "Starting download pass: %d documents, concurrency=%d",
            total, cfg.concurrency,
            extra={"total": total, "concurrency": cfg.concurrency},
        )

        semaphore = asyncio.Semaphore(cfg.concurrency)
        timeout = httpx.Timeout(cfg.timeout, connect=10.0)
        headers = {"User-Agent": "DTU-RAG-Downloader/1.0"}

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            tasks = [
                _download_one(semaphore, client, db, doc, cfg, shutdown)
                for doc in pending
            ]
            await asyncio.gather(*tasks)

        stats = db.get_stats().get(STAGE, {})
        log.info(
            "Download pass complete | done=%s failed=%s duplicate=%s",
            stats.get("done", 0),
            stats.get("failed", 0),
            stats.get("skipped", 0),  # mark_duplicate sets download_status='skipped'
            extra={"stats": stats},
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Stage 3 — download pending PDFs into data/raw/pdfs/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest-path", type=Path, default=Path("manifest.db"))
    parser.add_argument("--output-dir",    type=Path, default=Path("data/raw/pdfs"))
    parser.add_argument("--concurrency",   type=int,  default=4)
    parser.add_argument("--timeout",       type=float, default=30.0)
    parser.add_argument("--max-retries",   type=int,  default=3)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    return Config(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    asyncio.run(run(_parse_args()))
