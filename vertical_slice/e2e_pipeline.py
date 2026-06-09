#!/usr/bin/env python3
"""
DTU RAG -- Full end-to-end pipeline.

Stages
------
  1. CHUNK   pdf_parser.py + ir_chunker.py  ->  chunk dicts
  2. EMBED   OpenAI text-embedding-3-small  ->  1536-dim vectors
  3. INDEX   Qdrant (Docker)  ->  persistent cosine collection
  4. QUERY   Multi-variant retrieval + gpt-4o-mini synthesis

Qdrant must be running before this script is invoked:

    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \\
        -v qdrant_storage:/qdrant/storage qdrant/qdrant

Run from the repo root (Rag_V1/):

    python vertical_slice/e2e_pipeline.py \\
        --pdf   dtu-chatbot/data/raw/pdfs/BTech_2022_ordinance.pdf \\
        --url   "https://dtu.ac.in/Web/Academics/ordinance/BTech_2022_ordinance.pdf" \\
        --title "DTU B.Tech Ordinance 2022" \\
        --doc-type ordinance \\
        --date  2022-01-01

Re-running with the same PDF is safe -- upsert is idempotent.
Use --recreate to wipe the collection and rebuild from scratch.
Add custom questions with --query (repeatable).
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap -- make dtu-chatbot/ importable from anywhere
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve().parent   # vertical_slice/
_REPO_ROOT = _HERE.parent                      # Rag_V1/
_CHATBOT   = _REPO_ROOT / "dtu-chatbot"

for _p in [str(_CHATBOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ingestion.ir_chunker import chunk_document
from ingestion.pdf_parser import parse_pdf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
COLLECTION_NAME = "dtu_rag"
VECTOR_DIM      = 1536
TOP_K           = 3
TOP_DEDUP       = 5
EMBED_BATCH     = 100

DEFAULT_QUERIES = [
    "What is the minimum attendance required to appear in the end-term exam?",
    "What is the grading formula for the O (Outstanding) grade?",
    "What is the maximum duration to complete a B.Tech degree?",
    "What happens if I fail too many courses in my first year?",
    "Can I withdraw from a semester due to health reasons?",
]

SYSTEM_PROMPT = (
    "You are the DTU institutional assistant. "
    "Answer ONLY based on the provided context chunks. "
    "Do not use any external knowledge about DTU.\n\n"
    "Rules:\n"
    "- If the context contains relevant information, use it -- even when terminology "
    "differs from the question.\n"
    "- If the context genuinely lacks the information, say exactly: "
    "'I could not find this information in the available DTU documents.'\n"
    "- Never speculate beyond what is in the context.\n"
    "- Cite the section heading for every factual claim."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(title: str, char: str = "=") -> None:
    bar = char * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _chunk_to_point_id(chunk_id: str) -> str:
    """Derive a stable UUID from the sha256 chunk_id for Qdrant upsert idempotency."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed texts in batches of EMBED_BATCH, return list of vectors."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def _rewrite_query(client: OpenAI, query: str) -> list[str]:
    """Return [original] + 3 formal rewrites for multi-variant retrieval."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query rewriter for a DTU university document search system. "
                    "Rewrite the student query into 3 alternative phrasings using formal "
                    "Indian university academic terminology as it would appear in an official "
                    "university ordinance or regulation. "
                    "Return ONLY a JSON array of 3 strings. No explanation. No markdown."
                ),
            },
            {"role": "user", "content": f"Student query: {query}"},
        ],
    )
    try:
        rewrites = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, AttributeError):
        rewrites = []
    return [query] + rewrites


# ---------------------------------------------------------------------------
# Stage 1 -- Chunk
# ---------------------------------------------------------------------------


def stage_chunk(
    pdf_path: Path,
    url: str,
    title: str,
    doc_type: str,
    date_published: str | None,
) -> list[dict]:
    """Parse PDF to IR, chunk, return chunk dicts."""
    _banner("STAGE 1 -- CHUNK  (pdf_parser -> IR -> chunks)")
    t0 = time.perf_counter()

    doc    = parse_pdf(pdf_path, url=url, doc_type=doc_type,
                       title=title, date_published=date_published)
    chunks = chunk_document(doc)

    elapsed = time.perf_counter() - t0
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["block_type"]] = by_type.get(c["block_type"], 0) + 1

    toks = [c["token_count"] for c in chunks]
    print(f"  PDF          : {pdf_path.name}")
    print(f"  IR blocks    : {len(doc.blocks)}")
    print(f"  Chunks total : {len(chunks)}  {by_type}")
    if toks:
        print(f"  Token sizes  : min={min(toks)}  avg={sum(toks)//len(toks)}  max={max(toks)}")
    print(f"  Time         : {elapsed:.1f}s")
    return chunks


# ---------------------------------------------------------------------------
# Stage 2 -- Embed
# ---------------------------------------------------------------------------


def stage_embed(client: OpenAI, chunks: list[dict]) -> list[list[float]]:
    """Embed all chunks with text-embedding-3-small."""
    _banner("STAGE 2 -- EMBED  (OpenAI text-embedding-3-small)")
    t0 = time.perf_counter()

    texts   = [c["text"] for c in chunks]
    vectors = _embed_batch(client, texts)

    elapsed = time.perf_counter() - t0
    print(f"  Chunks embedded : {len(vectors)}")
    print(f"  Vector dim      : {len(vectors[0]) if vectors else 0}")
    print(f"  Time            : {elapsed:.1f}s")
    return vectors


# ---------------------------------------------------------------------------
# Stage 3 -- Index
# ---------------------------------------------------------------------------


def stage_index(
    chunks: list[dict],
    vectors: list[list[float]],
    qdrant: QdrantClient,
    recreate: bool = False,
) -> None:
    """Upsert chunks into Qdrant.

    Creates the collection if it doesn't exist.
    With recreate=True, drops and rebuilds it first (clean slate).
    Default upsert is idempotent -- re-indexing the same PDF is safe.
    """
    _banner("STAGE 3 -- INDEX  (Qdrant Docker)")
    t0 = time.perf_counter()

    existing = {c.name for c in qdrant.get_collections().collections}

    if recreate and COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
        existing.discard(COLLECTION_NAME)
        print(f"  Dropped existing collection '{COLLECTION_NAME}'")

    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"  Created collection '{COLLECTION_NAME}'  (dim={VECTOR_DIM}, cosine)")

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
            },
        )
        for c, vec in zip(chunks, vectors)
    ]

    # Upsert in batches to stay within gRPC message size limits
    batch_size = 256
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )

    info    = qdrant.get_collection(COLLECTION_NAME)
    elapsed = time.perf_counter() - t0
    print(f"  Collection      : {COLLECTION_NAME}")
    print(f"  Points total    : {info.points_count}")
    print(f"  Upserted now    : {len(points)}")
    print(f"  Time            : {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Stage 4 -- Query
# ---------------------------------------------------------------------------


def stage_query(
    client: OpenAI,
    qdrant: QdrantClient,
    queries: list[str],
    batch_year: int | None = None,
) -> None:
    """Multi-variant retrieval from Qdrant + gpt-4o-mini synthesis.

    batch_year: if set (e.g. 2022), only retrieve chunks from that ordinance year.
                0 / None means no filter — search the full corpus.
    """
    _banner("STAGE 4 -- QUERY  (multi-variant retrieval + gpt-4o-mini)")
    info   = qdrant.get_collection(COLLECTION_NAME)
    n_docs = info.points_count or 0

    # Build a Qdrant filter that matches both the requested year AND evergreen
    # content (batch_year == 0, e.g. HTML department pages, hostel info).
    qfilter: Filter | None = None
    if batch_year:
        qfilter = Filter(
            should=[
                FieldCondition(key="batch_year", match=MatchValue(value=batch_year)),
                FieldCondition(key="batch_year", match=MatchValue(value=0)),
            ]
        )

    for q_num, query in enumerate(queries, start=1):
        print(f"\n{'-' * 60}")
        print(f"  Q{q_num}: {query}")
        if batch_year:
            print(f"  Filter: batch_year={batch_year} OR batch_year=0")
        print(f"{'-' * 60}")

        variants  = _rewrite_query(client, query)
        q_vectors = _embed_batch(client, variants)

        # Retrieve and deduplicate across query variants
        best: dict[str, dict] = {}
        for vec in q_vectors:
            hits = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vec,
                query_filter=qfilter,
                limit=min(TOP_K, max(n_docs, 1)),
                with_payload=True,
            ).points
            for hit in hits:
                cid   = hit.payload["chunk_id"]
                score = hit.score   # Qdrant cosine: higher = more similar
                if cid not in best or score > best[cid]["score"]:
                    best[cid] = {"payload": hit.payload, "score": score}

        top = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:TOP_DEDUP]

        print("  Retrieved:")
        for i, chunk in enumerate(top, 1):
            p = chunk["payload"]
            print(
                f"    [{i}] score={chunk['score']:.3f}  "
                f"type={p.get('block_type','?')}  "
                f"page={p.get('page_number','?')}  "
                f"section={str(p.get('section_heading',''))[:55]}"
            )

        context_parts = [
            f"[Chunk {i} | Section: {c['payload']['section_heading']} "
            f"| Source: {c['payload']['source_url']}]\n{c['payload']['text']}"
            for i, c in enumerate(top, 1)
        ]
        context = "\n\n".join(context_parts)

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        print(f"\n  Answer:\n{resp.choices[0].message.content.strip()}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DTU RAG: chunk -> embed -> index (Qdrant) -> query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pdf",      required=True,  type=Path, help="Path to PDF file")
    ap.add_argument("--url",      required=True,             help="Source URL for the PDF")
    ap.add_argument("--title",    required=True,             help="Document title")
    ap.add_argument("--doc-type", default="ordinance",       help="Document type tag")
    ap.add_argument("--date",     default=None,              help="Publication date YYYY-MM-DD")
    ap.add_argument(
        "--qdrant-host", default="localhost",
        help="Qdrant host (default: localhost)",
    )
    ap.add_argument(
        "--qdrant-port", type=int, default=6333,
        help="Qdrant HTTP port (default: 6333)",
    )
    ap.add_argument(
        "--recreate", action="store_true", default=False,
        help="Drop and rebuild the Qdrant collection before indexing",
    )
    ap.add_argument(
        "--query-only", action="store_true", default=False,
        help="Skip chunk/embed/index -- query an existing collection",
    )
    ap.add_argument(
        "--query", action="append", dest="queries", metavar="Q",
        help="Question to ask (repeatable). Defaults to 5 built-in questions.",
    )
    ap.add_argument(
        "--batch-year", dest="batch_year", type=int, default=None,
        help="Filter retrieval to this ordinance batch year (e.g. 2022). "
             "Also includes evergreen content (batch_year=0). Default: no filter.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set.\n"
            "Copy vertical_slice/.env.example -> vertical_slice/.env and fill in your key."
        )

    if not args.query_only and not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    oai    = OpenAI(api_key=api_key)
    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    queries = args.queries or DEFAULT_QUERIES

    t_total = time.perf_counter()
    _banner("DTU RAG -- END-TO-END PIPELINE  (Qdrant)", char="*")
    print(f"  PDF      : {args.pdf}")
    print(f"  Title    : {args.title}")
    print(f"  Qdrant   : {args.qdrant_host}:{args.qdrant_port}")
    print(f"  Recreate : {args.recreate}")
    print(f"  Queries  : {len(queries)}")

    if not args.query_only:
        chunks  = stage_chunk(args.pdf, args.url, args.title, args.doc_type, args.date)
        vectors = stage_embed(oai, chunks)
        stage_index(chunks, vectors, qdrant, recreate=args.recreate)

    stage_query(oai, qdrant, queries, batch_year=args.batch_year)

    _banner(f"DONE  --  total time {time.perf_counter() - t_total:.1f}s", char="*")


if __name__ == "__main__":
    main()
