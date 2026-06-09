"""
DTU RAG pipeline -- Stage 3 PDF download worker.

Pulls every pending PDF URL from the manifest, streams each file to disk
with SHA-256 deduplication, and updates the manifest atomically.

Usage::

    python ingestion/download_worker.py \\
        --manifest-path manifest/manifest.db \\
        --output-dir    data/raw/pdfs \\
        --concurrency   4 \\
        --timeout       60 \\
        --total-timeout 300 \\
        --max-retries   3 \\
        --log-level     INFO
"""
from __future__ import annotations

import argparse
import asyncio
import enum
import errno
import hashlib
import json
import logging
import os
import signal
import sqlite3
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, TypedDict
from urllib.parse import urlparse

import httpx

from manifest.manifest import ManifestDB
from manifest.resumability import recover_and_get_pending

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE: Final[str] = "download"
STREAM_CHUNK: Final[int] = 65_536
NOTES_LIMIT: Final[int] = 1_000
MAX_STEM_LEN: Final[int] = 180
COLLISION_COUNTER_MAX: Final[int] = 100

# application/octet-stream is excluded by default — any binary blob carries
# that type.  Enable per-run with --allow-octet-stream for misconfigured hosts.
PDF_CONTENT_TYPES: Final[frozenset[str]] = frozenset({
    "application/pdf",
    "application/x-pdf",
})

_OCTET_STREAM_TYPES: Final[frozenset[str]] = frozenset({
    "application/octet-stream",
    "binary/octet-stream",
})

_SAFE_FILENAME_CHARS: Final[frozenset[str]] = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_."
)

# HTTP status codes worth retrying (server-side transient errors)
_RETRYABLE_HTTP_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Outcome strings -- module-level constants, never bare literals in bodies
# ---------------------------------------------------------------------------


class DownloadOutcome(str, enum.Enum):
    """Canonical outcome tokens written to logs and the manifest."""

    DONE             = "done"
    FAILED           = "failed"
    DUPLICATE        = "duplicate"
    SKIPPED_SHUTDOWN = "skipped_shutdown"
    TIMEOUT          = "timeout"
    CONNECTION_ERROR = "connection_error"
    OS_ERROR         = "os_error"
    UNEXPECTED_ERROR = "unexpected_error"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DownloadError(Exception):
    """Base class for all download-stage failures."""


class HttpError(DownloadError):
    """HTTP 4xx / 5xx response."""

    def __init__(self, url: str, status_code: int) -> None:
        super().__init__(f"HTTP {status_code} for {url!r}")
        self.status_code = status_code

    @property
    def is_retryable(self) -> bool:
        """True for transient server errors worth retrying."""
        return self.status_code in _RETRYABLE_HTTP_STATUS


class ContentTypeError(DownloadError):
    """Response Content-Type is not a recognised PDF MIME type."""

    def __init__(self, url: str, content_type: str) -> None:
        super().__init__(f"Unexpected Content-Type {content_type!r} for {url!r}")
        self.content_type = content_type


class EmptyBodyError(DownloadError):
    """Server returned a zero-byte body."""


class DiskFullError(DownloadError):
    """No space left on device while writing to disk."""


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingDocument:
    """Typed representation of a manifest row pending download."""

    url: str


# ---------------------------------------------------------------------------
# ManifestDB structural interface
# ---------------------------------------------------------------------------


class ManifestDBProtocol(Protocol):
    """Structural interface consumed by the download worker.

    Typed against this Protocol so the worker can be unit-tested with a stub
    without requiring a real SQLite file.
    """

    def mark_running(self, url: str, stage: str) -> None: ...
    def update_stage(
        self, url: str, stage: str, status: str, notes: str | None = None
    ) -> None: ...
    def get_document_by_hash(
        self,
        file_hash: str,
        exclude_url: str,
        confirmed_statuses: tuple[str, ...] = ("done",),
    ) -> sqlite3.Row | None: ...
    def mark_duplicate(self, url: str, canonical_url: str | None = None) -> None: ...
    def finalize_download(
        self, url: str, stage: str, path: Path, size: int, file_hash: str
    ) -> None: ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """All runtime tunables -- maps 1:1 to CLI flags."""

    manifest_path: Path = Path("manifest/manifest.db")
    output_dir: Path = Path("data/raw/pdfs")
    concurrency: int = 4
    timeout_seconds: float = 60.0
    total_timeout_seconds: float = 300.0
    max_retries: int = 3
    retry_min_wait: float = 2.0
    retry_max_wait: float = 30.0
    user_agent: str = "DTU-RAG-Downloader/1.0 (+https://dtu.ac.in)"
    allow_octet_stream: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate tunables at construction time."""
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency!r}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds!r}")
        if self.total_timeout_seconds <= 0:
            raise ValueError(
                f"total_timeout_seconds must be > 0, got {self.total_timeout_seconds!r}"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries!r}")


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


class LogFields(TypedDict, total=False):
    """Typed schema for structured log context fields emitted by this worker."""

    url: str
    outcome: str
    bytes_written: int
    elapsed_ms: int
    dest: str
    error: str
    canonical_url: str
    stage: str
    pending_count: int
    concurrency: int
    timeout_seconds: float
    manifest_path: str
    output_dir: str
    signal_name: str
    attempt: int
    retry_delay: float


# Derived once at import time from the TypedDict annotations so LogFields is
# the single source of truth -- no manual _LOG_FIELD_NAMES list to keep in sync.
_LOG_KEYS: Final[frozenset[str]] = frozenset(LogFields.__annotations__)


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the log record as a single JSON line."""
        payload: dict[str, str | int | float | bool | None] = {
            "ts":     self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key in _LOG_KEYS:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging(level_name: str = "INFO") -> None:
    """Wire up JSON structured logging to stdout at the requested level."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Shared download context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DownloadContext:
    """Immutable shared state threaded through every download coroutine."""

    client: httpx.AsyncClient
    db: ManifestDBProtocol
    config: Config
    shutdown_event: asyncio.Event
    allowed_content_types: frozenset[str]


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def _safe_stem(url: str) -> str:
    """Derive a filesystem-safe stem from a URL path component."""
    raw_stem = Path(urlparse(url).path).stem or "document"
    cleaned = "".join(c if c in _SAFE_FILENAME_CHARS else "_" for c in raw_stem)
    return (cleaned or "document")[:MAX_STEM_LEN]


def _collision_free_path(directory: Path, stem: str) -> Path:
    """Return a unique .pdf path in directory.

    Tries stem.pdf, stem_2.pdf ... up to COLLISION_COUNTER_MAX, then falls
    back to a UUID hex suffix to avoid O(N^2) scans on degenerate corpora.
    """
    candidate = directory / f"{stem}.pdf"
    if not candidate.exists():
        return candidate
    for counter in range(2, COLLISION_COUNTER_MAX + 1):
        candidate = directory / f"{stem}_{counter}.pdf"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}_{uuid.uuid4().hex[:8]}.pdf"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(t0: float) -> int:
    """Return elapsed milliseconds since t0 (monotonic clock)."""
    return int((time.monotonic() - t0) * 1000)


def _truncate_notes(exc: BaseException, prefix: str = "") -> str:
    """Format an exception as a manifest notes string, capped at NOTES_LIMIT."""
    msg = f"{prefix}: {exc}" if prefix else str(exc)
    return msg[:NOTES_LIMIT]


def _log_outcome(
    url: str,
    outcome: DownloadOutcome,
    bytes_written: int = 0,
    t0: float = 0.0,
    error: str | None = None,
    additional: LogFields | None = None,
) -> None:
    """Emit a structured log record for a completed (or failed) download."""
    fields: LogFields = {
        "url":           url,
        "outcome":       outcome.value,
        "bytes_written": bytes_written,
        "elapsed_ms":    _elapsed_ms(t0),
    }
    if error:
        fields["error"] = error
    if additional:
        # Guard against callers accidentally overwriting the four required keys.
        _required = {"url", "outcome", "bytes_written", "elapsed_ms"}
        safe_extra = {k: v for k, v in additional.items() if k not in _required}
        fields.update(safe_extra)
    level = (
        logging.WARNING
        if outcome in {
            DownloadOutcome.FAILED,
            DownloadOutcome.TIMEOUT,
            DownloadOutcome.CONNECTION_ERROR,
            DownloadOutcome.OS_ERROR,
            DownloadOutcome.UNEXPECTED_ERROR,
        }
        else logging.INFO
    )
    LOG.log(level, "download %s", outcome.value, extra=fields)


def _safe_update_stage(
    db: ManifestDBProtocol, url: str, status: str, notes: str
) -> bool:
    """Write a stage status update, logging on failure.

    Returns True on success, False if the manifest write itself raised.
    Callers can inspect the return value to decide whether to continue or abort.
    """
    try:
        db.update_stage(url, STAGE, status, notes=notes)
        return True
    except Exception:
        LOG.error(
            "Failed to write stage status to manifest.",
            extra={"url": url, "outcome": status},
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Core fetch-and-persist
# ---------------------------------------------------------------------------


async def _fetch_and_persist(url: str, ctx: _DownloadContext) -> None:
    """Stream-download url, deduplicate by SHA-256, and write to disk atomically.

    Uses mkstemp + atomic rename so no partial PDF ever appears in output_dir.
    Disk writes are offloaded via asyncio.to_thread to avoid blocking the loop
    on slow storage.

    Raises DownloadError (and subclasses) on all recoverable failures.
    CancelledError propagates unchanged after cleaning up the temp file.
    """
    async with ctx.client.stream("GET", url) as response:
        if response.status_code >= 400:
            raise HttpError(url, response.status_code)

        mime_type = (
            response.headers.get("content-type", "").split(";")[0].strip().lower()
        )
        if not mime_type or mime_type not in ctx.allowed_content_types:
            raise ContentTypeError(url, mime_type or "(empty)")

        tmp_fd, tmp_path_str = tempfile.mkstemp(
            suffix=".tmp", dir=ctx.config.output_dir
        )
        tmp_path = Path(tmp_path_str)
        hasher = hashlib.sha256()
        total_bytes = 0

        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                async for chunk in response.aiter_bytes(STREAM_CHUNK):
                    try:
                        await asyncio.to_thread(fh.write, chunk)
                    except OSError as exc:
                        if exc.errno == errno.ENOSPC:
                            raise DiskFullError(
                                f"Disk full writing temp file for {url!r}"
                            ) from exc
                        raise DownloadError(
                            f"I/O error writing temp file for {url!r}: {exc}"
                        ) from exc
                    hasher.update(chunk)
                    total_bytes += len(chunk)
        except BaseException:
            # Covers CancelledError (BaseException since Python 3.8) and all
            # DownloadError subclasses -- always clean up the temp file.
            tmp_path.unlink(missing_ok=True)
            raise

    if total_bytes == 0:
        tmp_path.unlink(missing_ok=True)
        raise EmptyBodyError(f"Empty body received for {url!r}")

    file_hash = hasher.hexdigest()

    duplicate_row: sqlite3.Row | None = ctx.db.get_document_by_hash(
        file_hash, exclude_url=url
    )
    if duplicate_row:
        tmp_path.unlink(missing_ok=True)
        ctx.db.mark_duplicate(url, canonical_url=duplicate_row["url"])
        _log_outcome(
            url,
            DownloadOutcome.DUPLICATE,
            additional={"canonical_url": duplicate_row["url"]},
        )
        return

    stem = _safe_stem(url)
    dest = _collision_free_path(ctx.config.output_dir, stem)
    try:
        tmp_path.rename(dest)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        if exc.errno == errno.ENOSPC:
            raise DiskFullError(f"Disk full renaming temp file to {dest}") from exc
        raise DownloadError(f"Could not rename temp file to {dest}: {exc}") from exc

    try:
        ctx.db.finalize_download(url, STAGE, dest, total_bytes, file_hash)
    except Exception:
        # Rename succeeded but the atomic DB commit failed.  Remove the orphaned
        # file so the next recovery cycle can retry from a clean state.
        # The manifest row stays 'running' and will be reset to 'pending' by
        # recover_and_get_pending on next startup.
        dest.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


async def _fetch_with_retry(url: str, ctx: _DownloadContext, t0: float) -> None:
    """Wrap _fetch_and_persist with exponential backoff on transient errors.

    Retries on connection errors, read timeouts, and retryable HTTP status
    codes (429, 5xx).  Non-retryable errors (404, ContentTypeError, etc.)
    propagate immediately.  A per-download wall-clock cap is enforced via
    asyncio.wait_for so a trickle server cannot hold a slot indefinitely.
    """
    delay = ctx.config.retry_min_wait

    for attempt in range(1, ctx.config.max_retries + 2):
        try:
            await asyncio.wait_for(
                _fetch_and_persist(url, ctx),
                timeout=ctx.config.total_timeout_seconds,
            )
            return
        except asyncio.TimeoutError as exc:
            raise DownloadError(
                f"Total timeout ({ctx.config.total_timeout_seconds}s) exceeded "
                f"for {url!r}"
            ) from exc
        except HttpError as exc:
            if not exc.is_retryable or attempt > ctx.config.max_retries:
                raise
            LOG.warning(
                "Retryable HTTP error — backing off.",
                extra={
                    "url": url, "attempt": attempt,
                    "error": str(exc), "retry_delay": delay,
                },
            )
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            if attempt > ctx.config.max_retries:
                raise DownloadError(
                    f"Max retries ({ctx.config.max_retries}) exceeded: {exc}"
                ) from exc
            LOG.warning(
                "Transient network error — backing off.",
                extra={
                    "url": url, "attempt": attempt,
                    "error": str(exc), "retry_delay": delay,
                },
            )

        await asyncio.sleep(delay)
        delay = min(delay * 2.0, ctx.config.retry_max_wait)


# ---------------------------------------------------------------------------
# Per-document worker coroutine
# ---------------------------------------------------------------------------


async def _download_one(
    doc: PendingDocument,
    ctx: _DownloadContext,
    sem: asyncio.Semaphore,
) -> None:
    """Download a single document, guarded by the shared semaphore."""
    url = doc.url

    if ctx.shutdown_event.is_set():
        _log_outcome(url, DownloadOutcome.SKIPPED_SHUTDOWN)
        return

    async with sem:
        if ctx.shutdown_event.is_set():
            _log_outcome(url, DownloadOutcome.SKIPPED_SHUTDOWN)
            return

        t0 = time.monotonic()
        LOG.debug("Download starting.", extra={"url": url})

        try:
            ctx.db.mark_running(url, STAGE)
            await _fetch_with_retry(url, ctx, t0)
            _log_outcome(url, DownloadOutcome.DONE, t0=t0)
        except DownloadError as exc:
            notes = _truncate_notes(exc)
            _log_outcome(url, DownloadOutcome.FAILED, t0=t0, error=notes)
            _safe_update_stage(ctx.db, url, DownloadOutcome.FAILED.value, notes)
        except httpx.TimeoutException as exc:
            notes = _truncate_notes(exc, prefix="Timeout")
            _log_outcome(url, DownloadOutcome.TIMEOUT, t0=t0, error=notes)
            _safe_update_stage(ctx.db, url, DownloadOutcome.FAILED.value, notes)
        except httpx.RequestError as exc:
            notes = _truncate_notes(exc, prefix=type(exc).__name__)
            _log_outcome(url, DownloadOutcome.CONNECTION_ERROR, t0=t0, error=notes)
            _safe_update_stage(ctx.db, url, DownloadOutcome.FAILED.value, notes)
        except OSError as exc:
            notes = _truncate_notes(exc, prefix="OS error")
            _log_outcome(url, DownloadOutcome.OS_ERROR, t0=t0, error=notes)
            _safe_update_stage(ctx.db, url, DownloadOutcome.FAILED.value, notes)
        except Exception as exc:
            notes = _truncate_notes(exc, prefix=type(exc).__name__)
            LOG.error(
                "Unexpected error during download.",
                extra={
                    "url": url,
                    "outcome": DownloadOutcome.UNEXPECTED_ERROR.value,
                    "bytes_written": 0,
                    "elapsed_ms": _elapsed_ms(t0),
                    "error": notes,
                },
                exc_info=True,
            )
            _safe_update_stage(ctx.db, url, DownloadOutcome.FAILED.value, notes)


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def run(config: Config) -> None:
    """Recover crashed rows, queue all pending PDFs, and download concurrently."""
    LOG.info(
        "PDF download worker starting.",
        extra={
            "manifest_path":   str(config.manifest_path),
            "output_dir":      str(config.output_dir),
            "concurrency":     config.concurrency,
            "timeout_seconds": config.timeout_seconds,
        },
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal(sig_num: int) -> None:
        LOG.info(
            "Signal received -- graceful shutdown initiated.",
            extra={"signal_name": signal.Signals(sig_num).name},
        )
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal, sig)
    except (NotImplementedError, AttributeError):
        # SIGTERM is not deliverable on Windows ProactorEventLoop; only SIGINT
        # is wired here.  loop.call_soon_threadsafe is required because
        # Event.set() acquires an internal lock and is not safe to call
        # directly from a signal handler on Python < 3.10.
        signal.signal(
            signal.SIGINT,
            lambda s, f: loop.call_soon_threadsafe(shutdown_event.set),
        )

    allowed_types = set(PDF_CONTENT_TYPES)
    if config.allow_octet_stream:
        allowed_types.update(_OCTET_STREAM_TYPES)

    with ManifestDB(config.manifest_path) as db:
        raw_rows = recover_and_get_pending(db, STAGE)
        pending = [PendingDocument(url=row["url"]) for row in raw_rows]
        LOG.info(
            "Crash recovery complete.",
            extra={"stage": STAGE, "pending_count": len(pending)},
        )
        if not pending:
            LOG.info("No pending documents -- nothing to do.", extra={"stage": STAGE})
            return

        http_timeout = httpx.Timeout(
            connect=10.0, read=config.timeout_seconds, write=10.0, pool=5.0
        )
        async with httpx.AsyncClient(
            timeout=http_timeout,
            headers={"User-Agent": config.user_agent},
            follow_redirects=True,
        ) as client:
            ctx = _DownloadContext(
                client=client,
                db=db,
                config=config,
                shutdown_event=shutdown_event,
                allowed_content_types=frozenset(allowed_types),
            )
            sem = asyncio.Semaphore(config.concurrency)
            tasks = [
                asyncio.create_task(
                    _download_one(doc, ctx, sem),
                    name=f"dl:{doc.url}",
                )
                for doc in pending
            ]
            results: list[None | BaseException] = await asyncio.gather(
                *tasks, return_exceptions=True
            )

        for task, result in zip(tasks, results):
            if isinstance(result, BaseException):
                LOG.error(
                    "Download task raised an unhandled exception.",
                    extra={"task": task.get_name(), "error": str(result)},
                    exc_info=(type(result), result, result.__traceback__),
                )

    LOG.info("PDF download worker finished.", extra={"stage": STAGE})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> Config:
    """Parse CLI arguments and return a validated Config."""
    defaults = Config()
    parser = argparse.ArgumentParser(
        prog="download_worker",
        description="DTU RAG pipeline - Stage 3 PDF download worker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=defaults.manifest_path,
        metavar="PATH", help="Path to the manifest SQLite database.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=defaults.output_dir,
        metavar="DIR", help="Directory to write downloaded PDFs into.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=defaults.concurrency,
        metavar="N", help="Maximum concurrent downloads (asyncio.Semaphore).",
    )
    parser.add_argument(
        "--timeout", type=float, default=defaults.timeout_seconds,
        dest="timeout_seconds", metavar="SEC",
        help="Per-chunk read timeout in seconds (httpx).",
    )
    parser.add_argument(
        "--total-timeout", type=float, default=defaults.total_timeout_seconds,
        dest="total_timeout_seconds", metavar="SEC",
        help="Per-download wall-clock timeout in seconds (asyncio.wait_for).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=defaults.max_retries,
        metavar="N", help="Maximum retry attempts on transient errors.",
    )
    parser.add_argument(
        "--user-agent", default=defaults.user_agent,
        metavar="UA", help="HTTP User-Agent header sent with every request.",
    )
    parser.add_argument(
        "--allow-octet-stream", action="store_true", default=False,
        help=(
            "Accept application/octet-stream as a valid PDF Content-Type. "
            "Disabled by default; enable only for servers known to misconfigure "
            "MIME types for legitimate PDFs."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=defaults.log_level,
        help="Logging verbosity level.",
    )
    args = parser.parse_args()
    return Config(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        total_timeout_seconds=args.total_timeout_seconds,
        max_retries=args.max_retries,
        user_agent=args.user_agent,
        allow_octet_stream=args.allow_octet_stream,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    _cfg = _parse_args()
    _configure_logging(_cfg.log_level)
    asyncio.run(run(_cfg))
