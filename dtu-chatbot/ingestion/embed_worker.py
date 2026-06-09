"""
DTU RAG pipeline -- Stage 6 embedding worker.

Reads chunk JSONL files for every manifest-tracked document where
chunk_status='done' and embed_status='pending', calls the OpenAI embeddings
API in batches, and writes per-document embedding JSONL files to
data/embeddings/.

Each output line is the original chunk dict with an 'embedding' key appended
(a list of floats). This file is what the index worker consumes.

Usage::

    python ingestion/embed_worker.py \\
        --manifest-path manifest/manifest.db \\
        --chunks-dir    data/chunks \\
        --output-dir    data/embeddings \\
        --model         text-embedding-3-small \\
        --batch-size    100 \\
        --log-level     INFO

Requires OPENAI_API_KEY in the environment (or a .env file at the project root
or dtu-chatbot/).
"""
from __future__ import annotations

import argparse
import enum
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

from openai import OpenAI

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).resolve().parent   # dtu-chatbot/ingestion/
_PROJ_ROOT = _HERE.parent                       # dtu-chatbot/

for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv
    # Try the dtu-chatbot root, then the repo root
    load_dotenv(_PROJ_ROOT / ".env") or load_dotenv(_PROJ_ROOT.parent / ".env")
except ImportError:
    pass

from manifest.manifest import ManifestDB
from manifest.queries import get_ready_to_embed
from manifest.resumability import recover_and_get_pending

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAGE: Final[str] = "embed"
NOTES_LIMIT: Final[int] = 1_000

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome tokens
# ---------------------------------------------------------------------------

class EmbedOutcome(str, enum.Enum):
    DONE              = "done"
    FAILED            = "failed"
    NO_CHUNKS         = "no_chunks"
    UNEXPECTED_ERROR  = "unexpected_error"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    manifest_path: Path = Path("manifest/manifest.db")
    chunks_dir:    Path = Path("data/chunks")
    output_dir:    Path = Path("data/embeddings")
    model:         str  = "text-embedding-3-small"
    batch_size:    int  = 100
    log_level:     str  = "INFO"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size!r}")


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class LogFields(TypedDict, total=False):
    url:              str
    outcome:          str
    chunks_embedded:  int
    elapsed_ms:       int
    dest:             str
    error:            str
    stage:            str
    pending_count:    int
    model:            str
    manifest_path:    str
    output_dir:       str
    chunks_path:      str


_LOG_KEYS: Final[frozenset[str]] = frozenset(LogFields.__annotations__)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
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
    level = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _truncate(exc: BaseException, prefix: str = "") -> str:
    msg = f"{prefix}: {exc}" if prefix else str(exc)
    return msg[:NOTES_LIMIT]


def _resolve_chunks_path(row, chunks_dir: Path) -> Path | None:
    """Return the chunk JSONL path for a manifest row, or None if not found.

    Preference order:
    1. chunk_notes — set by the chunk worker; may be an absolute or relative path.
    2. Convention: {chunks_dir}/{file_path_stem}_ir_chunks.jsonl  (html_parser path)
    3. Convention: {chunks_dir}/{file_path_stem}_chunks.jsonl     (legacy path)
    """
    notes = row["chunk_notes"]
    if notes:
        p = Path(notes)
        if p.exists():
            return p

    fp = row["file_path"]
    if fp:
        stem = Path(fp).stem
        for suffix in ("_ir_chunks.jsonl", "_chunks.jsonl"):
            p = chunks_dir / f"{stem}{suffix}"
            if p.exists():
                return p

    return None


def _load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _embed_all(client: OpenAI, chunks: list[dict], model: str, batch_size: int) -> list[dict]:
    """Embed chunks in batches; return new dicts with 'embedding' appended."""
    embedded: list[dict] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        resp  = client.embeddings.create(model=model, input=[c["text"] for c in batch])
        for chunk, item in zip(batch, resp.data):
            embedded.append({**chunk, "embedding": item.embedding})
    return embedded


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

def _process_one(row, client: OpenAI, db: ManifestDB, config: Config) -> None:
    url = row["url"]
    t0  = time.monotonic()

    chunks_path = _resolve_chunks_path(row, config.chunks_dir)
    if chunks_path is None:
        notes = "chunk file not found"
        db.update_stage(url, STAGE, "failed", notes=notes)
        LOG.warning(
            "Chunk file not found; skipping.",
            extra={"url": url, "outcome": EmbedOutcome.NO_CHUNKS.value, "error": notes},
        )
        return

    LOG.debug("Embedding document.", extra={"url": url, "chunks_path": str(chunks_path)})

    chunks = _load_chunks(chunks_path)
    if not chunks:
        notes = f"chunk file empty: {chunks_path}"
        db.update_stage(url, STAGE, "failed", notes=notes)
        LOG.warning(
            "Chunk file is empty; skipping.",
            extra={"url": url, "outcome": EmbedOutcome.NO_CHUNKS.value, "error": notes},
        )
        return

    embedded = _embed_all(client, chunks, config.model, config.batch_size)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Use the chunk file stem so the lineage is traceable:
    # abc123_ir_chunks.jsonl → abc123_ir_chunks_embeddings.jsonl
    out_path = config.output_dir / f"{chunks_path.stem}_embeddings.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for c in embedded:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    db.update_stage(url, STAGE, "done", notes=str(out_path))
    LOG.info(
        "Embed done.",
        extra={
            "url":             url,
            "outcome":         EmbedOutcome.DONE.value,
            "chunks_embedded": len(embedded),
            "dest":            str(out_path),
            "elapsed_ms":      _elapsed_ms(t0),
        },
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config: Config) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set. "
            "Set it in the environment or add it to a .env file."
        )

    LOG.info(
        "Embed worker starting.",
        extra={
            "manifest_path": str(config.manifest_path),
            "output_dir":    str(config.output_dir),
            "model":         config.model,
            "stage":         STAGE,
        },
    )

    client = OpenAI(api_key=api_key)

    with ManifestDB(config.manifest_path) as db:
        # Crash recovery: reset any running rows from a prior crash back to pending.
        recover_and_get_pending(db, STAGE)

        rows = get_ready_to_embed(db)
        LOG.info(
            "Documents ready to embed.",
            extra={"stage": STAGE, "pending_count": len(rows)},
        )

        if not rows:
            LOG.info("Nothing to embed.", extra={"stage": STAGE})
            return

        for row in rows:
            url = row["url"]
            try:
                db.mark_running(url, STAGE)
                _process_one(row, client, db, config)
            except Exception as exc:
                notes = _truncate(exc, prefix=type(exc).__name__)
                LOG.error(
                    "Unexpected error embedding document.",
                    extra={
                        "url":     url,
                        "outcome": EmbedOutcome.UNEXPECTED_ERROR.value,
                        "error":   notes,
                    },
                    exc_info=True,
                )
                try:
                    db.update_stage(url, STAGE, "failed", notes=notes)
                except Exception:
                    pass

    LOG.info("Embed worker finished.", extra={"stage": STAGE})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> Config:
    defaults = Config()
    ap = argparse.ArgumentParser(
        prog="embed_worker",
        description="DTU RAG pipeline - Stage 6 embedding worker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--manifest-path", type=Path, default=defaults.manifest_path,
        metavar="PATH", help="Path to the manifest SQLite database.",
    )
    ap.add_argument(
        "--chunks-dir", type=Path, default=defaults.chunks_dir,
        metavar="DIR", help="Directory containing chunk JSONL files.",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=defaults.output_dir,
        metavar="DIR", help="Directory to write embedding JSONL files into.",
    )
    ap.add_argument(
        "--model", default=defaults.model,
        help="OpenAI embeddings model name.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=defaults.batch_size,
        dest="batch_size", metavar="N",
        help="Number of texts per embeddings API call.",
    )
    ap.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=defaults.log_level,
    )
    args = ap.parse_args()
    return Config(
        manifest_path=args.manifest_path,
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        model=args.model,
        batch_size=args.batch_size,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    _cfg = _parse_args()
    _configure_logging(_cfg.log_level)
    run(_cfg)
