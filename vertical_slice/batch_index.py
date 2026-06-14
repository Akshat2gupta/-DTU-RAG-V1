#!/usr/bin/env python3
"""
DTU RAG -- Manifest-driven batch indexer.

Reads every downloaded PDF from manifest.db and runs it through:
    parse -> chunk -> embed -> Qdrant upsert

Skips PDFs already marked index_status='done' (resumable).

Run from the repo root (Rag_V1/):
    python vertical_slice/batch_index.py [--recreate]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF — used for scanned-PDF detection

# -- path bootstrap ----------------------------------------------------------
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_CHATBOT   = _REPO_ROOT / "dtu-chatbot"

for _p in [str(_CHATBOT), str(_REPO_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

from openai import OpenAI
from qdrant_client import QdrantClient

from manifest.manifest import ManifestDB
from e2e_pipeline import (
    COLLECTION_NAME,
    VECTOR_DIM,
    _chunk_to_point_id,
    _embed_batch,
)
from ingestion.ir_chunker import chunk_document
from ingestion.pdf_parser import parse_pdf
from qdrant_client.models import Distance, PointStruct, VectorParams

MANIFEST_DB = _CHATBOT / "manifest" / "manifest.db"

_SCANNED_THRESHOLD = 80  # chars per page below which a page is considered scanned


def _is_mostly_scanned(file_path: Path) -> bool:
    """True when fewer than half the pages have embedded text.
    Scanned PDFs belong to ocr_worker.py, not this script."""
    try:
        with fitz.open(str(file_path)) as doc:
            n = len(doc)
            if n == 0:
                return False
            with_text = sum(
                1 for page in doc
                if len(page.get_text("text").strip()) >= _SCANNED_THRESHOLD
            )
        return with_text < 0.5 * n
    except Exception:
        return False


def _derive_title(url: str, file_path: str) -> str:
    """Best-effort title from the URL filename (stable even when the local
    file was saved under a de-duplicated name like *_2.pdf)."""
    stem = Path(url.split("?")[0].rstrip("/")).stem or Path(file_path).stem
    return stem.replace("_", " ").replace("-", " ").strip()


def _process_pdf(
    oai: OpenAI,
    qdrant: QdrantClient,
    row: dict,
) -> tuple[int, float]:
    """Parse, chunk, embed, upsert one PDF. Returns (n_chunks, elapsed_s)."""
    t0 = time.perf_counter()
    url        = row["url"]
    fp_raw     = row["file_path"]
    file_path  = Path(fp_raw) if Path(fp_raw).is_absolute() else _CHATBOT / fp_raw
    doc_type   = row["document_type"] or "ordinance"
    date_pub   = row["date_published"] or ""
    title      = row["title"] or _derive_title(url, str(file_path))

    doc    = parse_pdf(file_path, url=url, doc_type=doc_type,
                       title=title, date_published=date_pub)
    chunks = chunk_document(doc)

    if not chunks:
        return 0, time.perf_counter() - t0

    texts   = [c["text"] for c in chunks]
    vectors = _embed_batch(oai, texts)

    points = [
        PointStruct(
            id=_chunk_to_point_id(c["chunk_id"]),
            vector=vec,
            payload={
                "chunk_id":        c["chunk_id"],
                "text":            c["text"],
                "section_heading": c["section_heading"],
                "source_url":      c["source_url"],
                "document_title":  c["document_title"],
                "document_type":   c["document_type"],
                "block_type":      c["block_type"],
                "token_count":     int(c.get("token_count") or 0),
                "page_number":     int(c.get("page_number") or 0),
                "batch_year":      int(c.get("batch_year") or 0),
                "chunk_index":     int(c.get("chunk_index") or 0),
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    for i in range(0, len(points), 256):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i:i+256])

    return len(chunks), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch index all downloaded PDFs into Qdrant")
    ap.add_argument("--recreate", action="store_true",
                    help="Drop and rebuild the Qdrant collection first")
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--reindex-done", action="store_true",
                    help="Re-process PDFs already marked index_status=done")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in vertical_slice/.env")

    oai    = OpenAI(api_key=api_key)
    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

    # Ensure collection exists
    existing = {c.name for c in qdrant.get_collections().collections}
    if args.recreate and COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
        existing.discard(COLLECTION_NAME)
        print(f"Dropped collection '{COLLECTION_NAME}'")
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")

    # Fetch eligible rows from manifest
    with ManifestDB(MANIFEST_DB) as db:
        cur = db._conn.execute(
            r"""SELECT * FROM documents
               WHERE download_status = 'done'
                 AND category LIKE '%\_pdf%' ESCAPE '\'
                 AND file_path IS NOT NULL
               ORDER BY id"""
        )
        all_rows = [dict(r) for r in cur.fetchall()]

    # Curated overrides: hand-verified docs in data/curated/ replace these
    # sources — machine-extracting them would re-add garbled chunks next to
    # the clean curated ones. Applies even with --reindex-done.
    from ingestion.curated_sources import curated_source_urls
    curated = curated_source_urls()
    for r in all_rows:
        if r["url"] in curated:
            print(f"SKIP (superseded by curated doc): {r['url']}")
    all_rows = [r for r in all_rows if r["url"] not in curated]

    if not args.reindex_done:
        rows = [r for r in all_rows if r.get("index_status") != "done"]
    else:
        rows = all_rows

    print(f"\nPDFs to index : {len(rows)}  (of {len(all_rows)} downloaded)")
    print(f"Collection    : {COLLECTION_NAME}  ({qdrant.get_collection(COLLECTION_NAME).points_count} points currently)")
    print(f"Qdrant        : {args.qdrant_host}:{args.qdrant_port}\n")

    if not rows:
        print("Nothing to do.")
        return

    total_chunks = 0
    t_total = time.perf_counter()

    for i, row in enumerate(rows, 1):
        fp = row.get("file_path") or ""
        if not fp:
            print(f"[{i}/{len(rows)}] SKIP  (no file_path recorded)")
            continue
        # Paths in manifest.db are relative to dtu-chatbot/
        resolved = Path(fp) if Path(fp).is_absolute() else _CHATBOT / fp
        if not resolved.exists():
            print(f"[{i}/{len(rows)}] SKIP  (file missing): {resolved}")
            continue

        fname = resolved.name

        # Scanned PDFs are ocr_worker.py's responsibility.
        if _is_mostly_scanned(resolved):
            print(f"[{i}/{len(rows)}] SKIP  (scanned — run ocr_worker.py): {fname}")
            continue

        print(f"[{i}/{len(rows)}] Processing: {fname}", flush=True)

        row["file_path"] = str(resolved)   # pass absolute path to _process_pdf
        try:
            n_chunks, elapsed = _process_pdf(oai, qdrant, row)
            total_chunks += n_chunks
            print(f"         -> {n_chunks} chunks  {elapsed:.1f}s")

            if n_chunks > 0:
                with ManifestDB(MANIFEST_DB) as db:
                    db.update_stage(row["url"], "chunk", "done")
                    db.update_stage(row["url"], "index", "done")
            else:
                print(f"         -> WARNING: 0 chunks — not marking as done")

        except Exception as exc:
            print(f"         -> ERROR: {exc}")
            with ManifestDB(MANIFEST_DB) as db:
                db.update_stage(row["url"], "index", "failed", notes=str(exc)[:500])

    # Self-healing: indexing raw ordinances re-adds chunks that the curated
    # docs superseded — retire them again automatically.
    if total_chunks > 0:
        from ingestion.chunk_retirement import retire_chunks
        retire_chunks(qdrant, COLLECTION_NAME)

    elapsed_total = time.perf_counter() - t_total
    final_count   = qdrant.get_collection(COLLECTION_NAME).points_count
    print(f"\nDone. {total_chunks} chunks upserted in {elapsed_total:.0f}s")
    print(f"Collection '{COLLECTION_NAME}' now has {final_count} points total.")


if __name__ == "__main__":
    main()
