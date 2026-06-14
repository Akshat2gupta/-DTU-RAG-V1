#!/usr/bin/env python3
"""
Index curated markdown documents into Qdrant.

Curated docs live in dtu-chatbot/data/curated/*.md — hand-verified content
(e.g. placement statistics transcribed from scanned T&P sheets that OCR
mangles).  Each '## ' section becomes one chunk; a 'Source: <url>' line inside
the section sets that chunk's source_url.

Run from repo root:
    python vertical_slice/index_curated.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import uuid
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_CHATBOT   = _REPO_ROOT / "dtu-chatbot"
_CURATED   = _CHATBOT / "data" / "curated"

for _p in (str(_CHATBOT), str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

COLLECTION_NAME = "dtu_rag"
EMBED_MODEL     = "text-embedding-3-small"

_SOURCE_RE     = re.compile(r"^Source:\s*(\S+)", re.MULTILINE)
_BATCH_YEAR_RE = re.compile(r"^Batch-year:\s*(\d{4})", re.MULTILINE)


def _chunk_id(doc_name: str, idx: int) -> str:
    return hashlib.sha256(f"curated::{doc_name}::{idx}".encode()).hexdigest()


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def split_sections(md: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) per '## ' section.

    The preamble (text before the first '## ') is NOT a standalone chunk —
    a title-plus-key chunk ranks high for every topical query but contains
    no data, so the LLM refuses. Instead it is appended to every section so
    each chunk is self-contained (branch codes resolve in-chunk)."""
    parts = re.split(r"^## ", md, flags=re.MULTILINE)
    preamble = parts[0].strip()
    sections: list[tuple[str, str]] = []
    for part in parts[1:]:
        lines = part.splitlines()
        body  = "## " + part.strip()
        if preamble:
            body += "\n\n" + preamble
        sections.append((lines[0].strip(), body))
    return sections


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set in vertical_slice/.env")

    oai    = OpenAI()
    qdrant = QdrantClient(host="localhost", port=6333)

    md_files = sorted(_CURATED.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No curated docs found in {_CURATED}")

    total = 0
    for md_path in md_files:
        md       = md_path.read_text(encoding="utf-8")
        doc_name = md_path.stem
        title    = md.splitlines()[0].lstrip("# ").strip()
        sections = split_sections(md)

        texts   = [f"{title} > {heading}\n\n{body}" for heading, body in sections]
        vectors = [
            r.embedding
            for r in oai.embeddings.create(model=EMBED_MODEL, input=texts).data
        ]

        points = []
        for i, ((heading, body), text, vec) in enumerate(zip(sections, texts, vectors)):
            m   = _SOURCE_RE.search(body)
            url = m.group(1) if m else "https://tnp.dtu.ac.in/"
            ym  = _BATCH_YEAR_RE.search(body)
            batch_year = int(ym.group(1)) if ym else 0   # 0 = evergreen
            cid = _chunk_id(doc_name, i)
            points.append(PointStruct(
                id=_point_id(cid),
                vector=vec,
                payload={
                    "chunk_id":        cid,
                    "text":            text,
                    "section_heading": heading,
                    "source_url":      url,
                    "document_title":  title,
                    "document_type":   "curated_stats",
                    "block_type":      "table",
                    "token_count":     len(text) // 4,
                    "page_number":     0,
                    "batch_year":      batch_year,
                    "chunk_index":     i,
                },
            ))
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
        print(f"{md_path.name}: {len(points)} chunks indexed")

    print(f"\nDone. {total} curated chunks. Collection now has "
          f"{qdrant.get_collection(COLLECTION_NAME).points_count} points.")


if __name__ == "__main__":
    main()
