#!/usr/bin/env python3
"""
HTML batch indexer — parses, chunks, embeds, and indexes all scraped HTML pages.

Targets manifest rows where:
  - category IN (dept_about, dept_faculty, hostel_html, saarthi_html, exam_html,
                  ordinance_html, notice_html, scholarship_html)
  - scrape_status = 'done'
  - file_path IS NOT NULL
  - index_notes != 'html_done'  (unless --force)

Run from Rag_V1/:
    python vertical_slice/html_batch_index.py
    python vertical_slice/html_batch_index.py --force
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_CHATBOT   = _REPO_ROOT / "dtu-chatbot"

for _p in (str(_CHATBOT), str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import re as _re

from manifest.manifest import ManifestDB
from ingestion.html_parser import parse_html_file
from ingestion.faculty_parser import parse_faculty_html
from ingestion.ir_chunker import chunk_document

_DEPT_RE = _re.compile(r"Departments/([^/]+)/faculty", _re.IGNORECASE)

MANIFEST_DB     = _CHATBOT / "manifest" / "manifest.db"
COLLECTION_NAME = "dtu_rag"
VECTOR_DIM      = 1536
EMBED_MODEL     = "text-embedding-3-small"
EMBED_BATCH     = 100

HTML_CATEGORIES = {
    # Academics UG
    "ordinance_html", "notice_html", "scholarship_html",
    "academics_php", "programme_html",
    # Academics PG
    "pg_academics_php",
    # Departments
    "dept_about", "dept_faculty", "dept_scheme", "dept_subpage",
    "faculty_profile",
    # About & administration
    "about_php", "admin_php",
    # R&D and institutional bodies
    "rnd_html", "nceet_html", "icc_html", "enggcell_html", "vigilance_html",
    # Governance
    "governance_php", "nirf_html",
    # Admissions
    "admissions_html", "saarthi_html",
    # Subdomains
    "exam_html", "hostel_html", "tnp_html", "library_html",
    # Student welfare & community
    "dsw_html", "community_html",
}


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def _chunk_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _ensure_collection(qdrant: QdrantClient) -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")


def _process_one(row: dict, oai: OpenAI, qdrant: QdrantClient) -> int:
    fp_raw    = row["file_path"]
    file_path = Path(fp_raw) if Path(fp_raw).is_absolute() else _CHATBOT / fp_raw
    if not file_path.exists():
        print(f"  SKIP: file not found: {file_path}")
        return 0

    if row.get("category") == "dept_faculty":
        m    = _DEPT_RE.search(row["url"])
        dept = m.group(1) if m else None
        doc  = parse_faculty_html(file_path, url=row["url"], dept=dept)
    else:
        doc = parse_html_file(file_path, url=row["url"], doc_type=row.get("category", "html"))

    chunks = chunk_document(doc)
    if not chunks:
        print(f"  -> 0 chunks")
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

    return len(chunks)


def main(force: bool = False, qdrant_host: str = "localhost", qdrant_port: int = 6333) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in vertical_slice/.env")

    oai    = OpenAI(api_key=api_key)
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    _ensure_collection(qdrant)

    before = qdrant.get_collection(COLLECTION_NAME).points_count

    with ManifestDB(MANIFEST_DB) as db:
        placeholders = ",".join("?" * len(HTML_CATEGORIES))
        cur = db._conn.execute(
            f"""SELECT * FROM documents
                WHERE scrape_status = 'done'
                  AND category IN ({placeholders})
                  AND file_path IS NOT NULL
                ORDER BY id""",
            list(HTML_CATEGORIES),
        )
        all_rows = [dict(r) for r in cur.fetchall()]

    rows = all_rows if force else [r for r in all_rows if r.get("index_notes") != "html_done"]

    print(f"\nHTML batch indexer")
    print(f"Pages to index  : {len(rows)}  (of {len(all_rows)} scraped)")
    print(f"Collection      : {COLLECTION_NAME}  ({before} points before)\n")

    if not rows:
        print("Nothing to do.")
        return

    total = 0
    t0    = time.perf_counter()

    for i, row in enumerate(rows, 1):
        label = row["url"].replace("https://", "").replace("http://", "")[:70]
        print(f"[{i}/{len(rows)}] {label}", flush=True)
        try:
            n = _process_one(row, oai, qdrant)
            total += n
            print(f"  -> {n} chunks")
            if n > 0:
                with ManifestDB(MANIFEST_DB) as db:
                    db._conn.execute(
                        "UPDATE documents SET index_notes=?, chunk_status=?, embed_status=?, index_status=? WHERE url=?",
                        ("html_done", "done", "done", "done", row["url"]),
                    )
                    db._conn.commit()
        except Exception as exc:
            print(f"  ERROR: {exc}")
            with ManifestDB(MANIFEST_DB) as db:
                db._conn.execute(
                    "UPDATE documents SET index_status=?, index_notes=? WHERE url=?",
                    ("failed", str(exc)[:500], row["url"]),
                )
                db._conn.commit()

    after = qdrant.get_collection(COLLECTION_NAME).points_count
    print(
        f"\nDone. {total} chunks in {time.perf_counter() - t0:.0f}s\n"
        f"Collection '{COLLECTION_NAME}': {before} -> {after} points"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",       action="store_true")
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    args = ap.parse_args()
    main(force=args.force, qdrant_host=args.qdrant_host, qdrant_port=args.qdrant_port)
