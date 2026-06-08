#!/usr/bin/env python3
"""
DTU RAG — Full end-to-end pipeline.

Stages
------
  1. CHUNK   pdf_parser.py + ir_chunker.py  →  chunk dicts
  2. EMBED   OpenAI text-embedding-3-small  →  1536-dim vectors
  3. INDEX   ChromaDB (ephemeral, rebuilt each run)
  4. QUERY   Multi-variant retrieval + gpt-4o-mini synthesis

Run from the repo root (Rag_V1/):

    python vertical_slice/e2e_pipeline.py \\
        --pdf  dtu-chatbot/data/raw/pdfs/BTech_2022_ordinance.pdf \\
        --url  "https://dtu.ac.in/Web/Academics/ordinance/BTech_2022_ordinance.pdf" \\
        --title "DTU B.Tech Ordinance 2022" \\
        --doc-type ordinance \\
        --date 2022-01-01

Add custom questions with --query (repeatable).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — make dtu-chatbot/ importable from anywhere
# ---------------------------------------------------------------------------

_HERE       = Path(__file__).resolve().parent          # vertical_slice/
_REPO_ROOT  = _HERE.parent                             # Rag_V1/
_CHATBOT    = _REPO_ROOT / "dtu-chatbot"

for _p in [str(_CHATBOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

import chromadb
from openai import OpenAI

from ingestion.pdf_parser  import parse_pdf
from ingestion.ir_chunker  import chunk_document

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
COLLECTION_NAME = "dtu_e2e"
TOP_K           = 3
TOP_DEDUP       = 5
EMBED_BATCH     = 100          # OpenAI recommends ≤ 2048 items; 100 is safe

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
    "- If the context contains relevant information, use it — even when terminology "
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


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed texts in batches of EMBED_BATCH, return list of vectors."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def _rewrite_query(client: OpenAI, query: str) -> list[str]:
    """Return [original] + 3 formal rewrites."""
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
# Stage 1 — Chunk
# ---------------------------------------------------------------------------

def stage_chunk(
    pdf_path: Path,
    url: str,
    title: str,
    doc_type: str,
    date_published: str,
) -> list[dict]:
    _banner("STAGE 1 — CHUNK  (pdf_parser → IR → chunks)")
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
# Stage 2 — Embed
# ---------------------------------------------------------------------------

def stage_embed(client: OpenAI, chunks: list[dict]) -> list[list[float]]:
    _banner("STAGE 2 — EMBED  (OpenAI text-embedding-3-small)")
    t0 = time.perf_counter()

    texts   = [c["text"] for c in chunks]
    vectors = _embed_batch(client, texts)

    elapsed = time.perf_counter() - t0
    print(f"  Chunks embedded : {len(vectors)}")
    print(f"  Vector dim      : {len(vectors[0]) if vectors else 0}")
    print(f"  Time            : {elapsed:.1f}s")
    return vectors


# ---------------------------------------------------------------------------
# Stage 3 — Index
# ---------------------------------------------------------------------------

def stage_index(
    chunks: list[dict],
    vectors: list[list[float]],
) -> chromadb.Collection:
    _banner("STAGE 3 — INDEX  (ChromaDB ephemeral)")
    t0 = time.perf_counter()

    chroma     = chromadb.EphemeralClient()
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids        = [c["chunk_id"] for c in chunks],
        embeddings = vectors,
        documents  = [c["text"]     for c in chunks],
        metadatas  = [
            {
                "section_heading": c["section_heading"],
                "source_url":      c["source_url"],
                "document_title":  c["document_title"],
                "document_type":   c["document_type"],
                "block_type":      c["block_type"],
                "token_count":     int(c.get("token_count") or 0),
                "page_number":     int(c.get("page_number") or 0),
            }
            for c in chunks
        ],
    )

    elapsed = time.perf_counter() - t0
    print(f"  Collection      : {COLLECTION_NAME}")
    print(f"  Documents       : {collection.count()}")
    print(f"  Time            : {elapsed:.1f}s")
    return collection


# ---------------------------------------------------------------------------
# Stage 4 — Query
# ---------------------------------------------------------------------------

def stage_query(
    client: OpenAI,
    collection: chromadb.Collection,
    queries: list[str],
) -> None:
    _banner("STAGE 4 — QUERY  (multi-variant retrieval + gpt-4o-mini)")
    n_docs = collection.count()

    for q_num, query in enumerate(queries, start=1):
        print(f"\n{'─' * 60}")
        print(f"  Q{q_num}: {query}")
        print(f"{'─' * 60}")

        # Rewrite + embed all variants
        variants  = _rewrite_query(client, query)
        q_vectors = _embed_batch(client, variants)

        # Retrieve and deduplicate across variants
        best: dict[str, dict] = {}
        for vec in q_vectors:
            results = collection.query(
                query_embeddings=[vec],
                n_results=min(TOP_K, n_docs),
            )
            for doc, meta, dist, cid in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                score = 1.0 - dist
                if cid not in best or score > best[cid]["score"]:
                    best[cid] = {"doc": doc, "meta": meta, "score": score}

        top = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:TOP_DEDUP]

        print("  Retrieved:")
        for i, chunk in enumerate(top, 1):
            btype = chunk["meta"].get("block_type", "prose")
            pg    = chunk["meta"].get("page_number", "?")
            print(
                f"    [{i}] score={chunk['score']:.2f}  "
                f"type={btype}  page={pg}  "
                f"section={chunk['meta']['section_heading'][:55]}"
            )

        # LLM synthesis
        context_parts = [
            f"[Chunk {i} | Section: {c['meta']['section_heading']} "
            f"| Source: {c['meta']['source_url']}]\n{c['doc']}"
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
        answer = resp.choices[0].message.content.strip()
        print(f"\n  Answer:\n{answer}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DTU RAG full pipeline: chunk → embed → index → query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pdf",      required=True,  type=Path, help="Path to PDF file")
    ap.add_argument("--url",      required=True,             help="Source URL for the PDF")
    ap.add_argument("--title",    required=True,             help="Document title")
    ap.add_argument("--doc-type", default="ordinance",       help="Document type")
    ap.add_argument("--date",     default=None,              help="Publication date YYYY-MM-DD")
    ap.add_argument(
        "--query", action="append", dest="queries", metavar="QUERY",
        help="Question to ask (repeatable). Defaults to 5 built-in questions.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set.\n"
            "Copy vertical_slice/.env.example → vertical_slice/.env and fill in your key."
        )

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    client  = OpenAI(api_key=api_key)
    queries = args.queries or DEFAULT_QUERIES

    t_total = time.perf_counter()

    _banner("DTU RAG — END-TO-END PIPELINE", char="█")
    print(f"  PDF      : {args.pdf}")
    print(f"  Title    : {args.title}")
    print(f"  Queries  : {len(queries)}")

    chunks     = stage_chunk(args.pdf, args.url, args.title, args.doc_type, args.date)
    vectors    = stage_embed(client, chunks)
    collection = stage_index(chunks, vectors)
    stage_query(client, collection, queries)

    _banner(f"DONE  —  total time {time.perf_counter() - t_total:.1f}s", char="█")


if __name__ == "__main__":
    main()
