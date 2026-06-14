#!/usr/bin/env python3
"""
OCR worker — Tesseract OCR on scanned PDFs via PyMuPDF, then chunk/embed/index.

Targets every downloaded PDF in the manifest where pdfplumber can't extract
enough text (the "likely scanned" path that batch_index.py skips over).

Usage (from Rag_V1/):
    python dtu-chatbot/ingestion/ocr_worker.py
    python dtu-chatbot/ingestion/ocr_worker.py --dpi 250 --lang eng
    python dtu-chatbot/ingestion/ocr_worker.py --force   # reprocess even done rows

Requires:
    - Tesseract 5.x installed (C:/Program Files/Tesseract-OCR/tesseract.exe)
    - OPENAI_API_KEY in vertical_slice/.env
    - Qdrant running on localhost:6333
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# ── path bootstrap ──────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent   # dtu-chatbot/ingestion/
_CHATBOT   = _HERE.parent                       # dtu-chatbot/
_REPO_ROOT = _CHATBOT.parent                    # Rag_V1/

for _p in (str(_CHATBOT), str(_REPO_ROOT), str(_REPO_ROOT / "vertical_slice")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Tesseract must be on PATH and TESSDATA_PREFIX must point to its data dir.
_TESS_BIN  = Path("C:/Program Files/Tesseract-OCR")
if _TESS_BIN.exists():
    os.environ.setdefault("TESSDATA_PREFIX", str(_TESS_BIN / "tessdata"))
    _path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(_TESS_BIN) not in _path_entries:
        os.environ["PATH"] = str(_TESS_BIN) + os.pathsep + os.environ.get("PATH", "")

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / "vertical_slice" / ".env")

import fitz                       # PyMuPDF
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from manifest.manifest import ManifestDB
from ingestion.classifier import classify_page
from ingestion.document_ir import Document, Heading, Paragraph
from ingestion.ir_chunker import chunk_document

# ── constants ────────────────────────────────────────────────────────────────
MANIFEST_DB     = _CHATBOT / "manifest" / "manifest.db"
COLLECTION_NAME = "dtu_rag"
VECTOR_DIM      = 1536
EMBED_MODEL     = "text-embedding-3-small"
EMBED_BATCH     = 100

# A page is "scanned" if pdfplumber-style text extraction gives fewer chars.
_SCANNED_THRESHOLD = 80

# Simple heading heuristics for OCR text (no font-size data available).
_NUMBERED_RE   = re.compile(r"^\d+[\.\)]\s+\S", re.ASCII)
_ALL_CAPS_WORD = re.compile(r"^[A-Z][A-Z\s\-/]{4,}$")


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    manifest_path: Path  = MANIFEST_DB
    qdrant_host:   str   = "localhost"
    qdrant_port:   int   = 6333
    dpi:           int   = 200
    lang:          str   = "eng"
    force:         bool  = False   # reprocess PDFs already marked index_status=ocr_done


# ── heading detection from raw OCR text ──────────────────────────────────────

def _is_ocr_heading(line: str, next_line: str) -> bool:
    """Lightweight heading detector for OCR'd government document text."""
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if line[-1] in ".,;":
        return False
    if _NUMBERED_RE.match(line):
        return True
    if _ALL_CAPS_WORD.match(line) and len(line) >= 5:
        return True
    # Short title-case line followed by a longer line
    if (len(line) <= 60 and line[0].isupper()
            and not re.search(r"\b(the|and|of|in|for|to|a|an|is|are|has|was)\b", line, re.I)
            and len(next_line.strip()) > len(line)):
        return True
    return False


def _build_ir_from_ocr(pages_text: list[tuple[int, str]], doc_meta: dict) -> Document:
    """
    Convert a list of (page_num, ocr_text) pairs into a Document IR.

    Heading detection is heuristic (short lines, numbered, ALL CAPS).
    Everything else becomes a Paragraph.  Each page that has content gets at
    least one Paragraph so no information is silently dropped.
    """
    blocks = []
    for page_num, text in pages_text:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        i = 0
        prose_buf: list[str] = []

        while i < len(lines):
            line = lines[i].strip()
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

            if _is_ocr_heading(line, next_line):
                if prose_buf:
                    blocks.append(Paragraph(" ".join(prose_buf), page=page_num))
                    prose_buf = []
                blocks.append(Heading(text=line, level=1, page=page_num))
            else:
                prose_buf.append(line)
            i += 1

        if prose_buf:
            blocks.append(Paragraph(" ".join(prose_buf), page=page_num))

    return Document(
        url=doc_meta["url"],
        title=doc_meta["title"] or Path(doc_meta.get("file_path", "")).stem,
        source_format="pdf_ocr",
        doc_type=doc_meta.get("document_type") or "unknown",
        date_published=doc_meta.get("date_published"),
        blocks=blocks,
    )


# ── per-page OCR ─────────────────────────────────────────────────────────────

def _extract_with_ocr(pdf_path: Path, dpi: int, lang: str) -> list[tuple[int, str]]:
    """
    Return (page_num, text) for every page that has meaningful content.

    Pages whose embedded text is already sufficient are used as-is (fast path).
    Pages below the scanned threshold get Tesseract OCR.
    """
    results: list[tuple[int, str]] = []
    skipped_by_class: dict[str, int] = {}

    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            # Fast path: page already has embedded text.
            embedded = page.get_text("text").strip()
            if len(embedded) >= _SCANNED_THRESHOLD:
                text = embedded
            else:
                # Scanned page — run Tesseract OCR.
                try:
                    tp   = page.get_textpage_ocr(flags=0, language=lang, dpi=dpi, full=True)
                    text = page.get_text(textpage=tp).strip()
                except Exception as exc:
                    print(f"    Page {i}: OCR failed ({exc})")
                    continue

            if len(text) < _SCANNED_THRESHOLD:
                continue

            # Same page-type gate as the pdfplumber pipeline: syllabus grids,
            # contact sheets and blank forms only add noise as OCR prose.
            page_type = classify_page(text)
            if page_type in ("syllabus", "contact", "form", "skip"):
                skipped_by_class[page_type] = skipped_by_class.get(page_type, 0) + 1
                continue

            results.append((i, text))

    if skipped_by_class:
        print(f"    Pages skipped by classifier: {skipped_by_class}")
    return results


def _is_mostly_scanned(pdf_path: Path) -> bool:
    """True when fewer than half the pages carry embedded text — i.e. the
    document needs OCR. Text PDFs are batch_index.py's job; re-indexing them
    here would overwrite its richer chunks (same chunk ids) with OCR-IR ones."""
    with fitz.open(str(pdf_path)) as doc:
        n = len(doc)
        if n == 0:
            return False
        with_text = sum(
            1 for page in doc if len(page.get_text("text").strip()) >= _SCANNED_THRESHOLD
        )
    return with_text < 0.5 * n


# ── embedding ────────────────────────────────────────────────────────────────

def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def _chunk_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


# ── ensure collection ────────────────────────────────────────────────────────

def _ensure_collection(qdrant: QdrantClient) -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'")


# ── per-document processing ───────────────────────────────────────────────────

def _process_one(row: dict, oai: OpenAI, qdrant: QdrantClient, cfg: Config) -> int:
    """OCR, chunk, embed, upsert one PDF. Returns number of chunks indexed."""
    fp_raw    = row["file_path"]
    file_path = Path(fp_raw) if Path(fp_raw).is_absolute() else _CHATBOT / fp_raw
    if not file_path.exists():
        print(f"  SKIP: file not found: {file_path}")
        return 0

    if not _is_mostly_scanned(file_path):
        print(f"  SKIP: {file_path.name} is a text PDF (batch_index.py handles it)")
        return 0

    print(f"  OCR scanning: {file_path.name}  (dpi={cfg.dpi})", flush=True)
    t0 = time.perf_counter()

    pages_text = _extract_with_ocr(file_path, cfg.dpi, cfg.lang)
    if not pages_text:
        print(f"  -> 0 pages with usable text after OCR")
        return 0

    print(f"  -> {len(pages_text)} pages with text, building IR...")
    doc    = _build_ir_from_ocr(pages_text, dict(row))
    chunks = chunk_document(doc)

    if not chunks:
        print(f"  -> 0 chunks produced")
        return 0

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
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i : i + 256])

    elapsed = time.perf_counter() - t0
    print(f"  -> {len(chunks)} chunks  {elapsed:.1f}s")
    return len(chunks)


# ── main ──────────────────────────────────────────────────────────────────────

def main(cfg: Config) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in vertical_slice/.env")

    oai    = OpenAI(api_key=api_key)
    qdrant = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
    _ensure_collection(qdrant)

    before = qdrant.get_collection(COLLECTION_NAME).points_count

    with ManifestDB(cfg.manifest_path) as db:
        cur = db._conn.execute(
            """SELECT * FROM documents
               WHERE download_status = 'done'
                 AND category LIKE '%_pdf%'
                 AND file_path IS NOT NULL
               ORDER BY id"""
        )
        all_rows = [dict(r) for r in cur.fetchall()]

    # Use runtime scan detection — the manifest is_ocr flag can be stale or wrong.
    def _resolve(r: dict) -> Path:
        fp = r["file_path"]
        return Path(fp) if Path(fp).is_absolute() else _CHATBOT / fp

    # Curated overrides: hand-verified docs in data/curated/ replace these
    # sources — re-OCRing them would re-add garbled chunks next to the clean
    # curated ones. Applies even with --force.
    from ingestion.curated_sources import curated_source_urls
    curated = curated_source_urls()
    superseded = [r for r in all_rows if r["url"] in curated]
    for r in superseded:
        print(f"SKIP (superseded by curated doc): {r['url']}")
    all_rows = [r for r in all_rows if r["url"] not in curated]

    scanned = [r for r in all_rows if _is_mostly_scanned(_resolve(r))]
    if cfg.force:
        rows = scanned
    else:
        rows = [r for r in scanned if r.get("index_notes") != "ocr_done"]

    print(f"\nOCR worker starting")
    print(f"PDFs to process : {len(rows)}  (of {len(all_rows)} downloaded)")
    print(f"Collection      : {COLLECTION_NAME}  ({before} points before)")
    print(f"DPI / lang      : {cfg.dpi} / {cfg.lang}\n")

    if not rows:
        print("Nothing to do.")
        return

    total_chunks = 0
    t_total = time.perf_counter()

    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['file_path']}")
        try:
            n = _process_one(row, oai, qdrant, cfg)
            total_chunks += n
            with ManifestDB(cfg.manifest_path) as db:
                if n > 0:
                    db.update_stage(row["url"], "chunk", "done")
                    db.update_stage(row["url"], "embed",  "done")
                    db.update_stage(row["url"], "index",  "done")
                # Mark ocr_done AFTER the update_stage calls — update_stage
                # rewrites index_notes (None when no notes arg), which would
                # wipe this marker and cause a full re-OCR on every run.
                db._conn.execute(
                    "UPDATE documents SET is_ocr=1, index_notes=? WHERE url=?",
                    ("ocr_done", row["url"]),
                )
                db._conn.commit()
        except Exception as exc:
            print(f"  ERROR: {exc}")

    after = qdrant.get_collection(COLLECTION_NAME).points_count
    print(
        f"\nDone. {total_chunks} new chunks in {time.perf_counter() - t_total:.0f}s\n"
        f"Collection '{COLLECTION_NAME}': {before} -> {after} points"
    )

    # Self-healing: re-OCRing ordinances re-adds chunks that the curated docs
    # superseded — retire them again automatically.
    if total_chunks > 0:
        from ingestion.chunk_retirement import retire_chunks
        retire_chunks(qdrant, COLLECTION_NAME)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> Config:
    ap = argparse.ArgumentParser(description="OCR worker for scanned PDFs")
    ap.add_argument("--manifest-path", type=Path, default=MANIFEST_DB)
    ap.add_argument("--qdrant-host",   default="localhost")
    ap.add_argument("--qdrant-port",   type=int, default=6333)
    ap.add_argument("--dpi",           type=int, default=200,
                    help="Tesseract rendering DPI (150–300; higher = slower but more accurate)")
    ap.add_argument("--lang",          default="eng",
                    help="Tesseract language code (default: eng)")
    ap.add_argument("--force",         action="store_true",
                    help="Re-process PDFs already marked index_notes=ocr_done")
    args = ap.parse_args()
    return Config(
        manifest_path=args.manifest_path,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        dpi=args.dpi,
        lang=args.lang,
        force=args.force,
    )


if __name__ == "__main__":
    main(_parse_args())
