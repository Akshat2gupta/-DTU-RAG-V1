"""
DTU RAG vertical slice — embed, index, query, inspect.

Usage:
    # Use the default hand-curated chunks (original behaviour)
    python vertical_slice/test_rag.py

    # Use chunks produced by the automated chunker (JSONL)
    python vertical_slice/test_rag.py --chunks ../dtu-chatbot/data/chunks/myfile_chunks.jsonl

    # Custom queries on any chunks file
    python vertical_slice/test_rag.py \
        --chunks ../dtu-chatbot/data/chunks/myfile_chunks.jsonl \
        --query "Can I repeat a year?" \
        --query "What is the hostel fee?"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
COLLECTION_NAME = "dtu_chunks"
TOP_K           = 3
TOP_DEDUP       = 5  # unique chunks to pass to LLM after deduplication

DEFAULT_CHUNKS_PATH = Path(__file__).parent / "chunks.json"

DEFAULT_QUERIES = [
    "My professor gave me wrong marks in internal assessment?",
    "How long can I take to finish my BTech degree?",
    "Can I get my semester cancelled due to health issues?",
]

SYSTEM_PROMPT = (
    """
You are the DTU institutional assistant.
Answer ONLY based on the provided context chunks.
Do not use any external knowledge about DTU.

Rules:
- If the context contains relevant information even if the
  terminology differs from the question, use it to answer.
  Example: if a student asks about "taking a break" and the
  context discusses "withdrawal from semester", explain that
  the formal process is called withdrawal and describe it.
- If the context genuinely does not contain relevant
  information, say exactly:
  "I could not find this information in the available DTU documents."
- Never speculate beyond what is in the context.
- Cite the section heading for every factual claim.
"""
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DTU RAG vertical slice")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks file (.json list or .jsonl from the chunker). "
             f"Defaults to {DEFAULT_CHUNKS_PATH.name}",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        metavar="QUERY",
        help="Query to run (repeatable). Defaults to 3 built-in demo queries.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from a .json list file or a .jsonl file (one JSON object per line)."""
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            chunks = [json.loads(line) for line in f if line.strip()]
        else:
            chunks = json.load(f)

    if not chunks:
        raise ValueError(f"No chunks found in {path}")

    return chunks


def rewrite_query(client: OpenAI, query: str) -> list[str]:
    """Return [original] + 3 formal rewrites of the query."""
    response = client.chat.completions.create(
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
                    "Return ONLY a JSON array of 3 strings. "
                    "No explanation. No markdown. Just the array. "
                    'Example: ["formal phrasing 1", "formal phrasing 2", "formal phrasing 3"]'
                ),
            },
            {"role": "user", "content": f"Student query: {query}"},
        ],
    )
    try:
        rewrites = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        rewrites = []
    return [query] + rewrites


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    queries = args.queries or DEFAULT_QUERIES

    # STEP 1 — Load chunks
    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks from {args.chunks.name}")

    # STEP 2 — Embed and index
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set — copy vertical_slice/.env.example to "
            "vertical_slice/.env and add your key."
        )
    client = OpenAI(api_key=api_key)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embed_response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embeddings = [item.embedding for item in embed_response.data]

    chroma = chromadb.EphemeralClient()
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "section_heading": c["section_heading"],
                "source_url":      c["source_url"],
                "document_title":  c["document_title"],
                "document_type":   c["document_type"],
                "date_published":  c["date_published"],
                # stored so the inspection dump below can show it
                "token_count":     int(c.get("token_count") or 0),
            }
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks\n")

    # STEP 2b — Inspect: print every chunk in the collection before querying
    stored = collection.get(include=["documents", "metadatas"])
    print("=" * 80)
    print(f"CHUNKS IN COLLECTION '{COLLECTION_NAME}': {len(stored['ids'])}")
    print("=" * 80)
    for cid, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        snippet = doc.replace("\n", " ")[:300]
        print(f"\nchunk_id       : {cid}")
        print(f"section_heading: {meta.get('section_heading')}")
        print(f"source_url     : {meta.get('source_url')}")
        print(f"token_count    : {meta.get('token_count')}")
        print(f"text[:300]     : {snippet}")
    print("\n" + "=" * 80 + "\n")

    # STEP 3 — Run queries
    for query_num, query in enumerate(queries, start=1):
        all_queries = rewrite_query(client, query)
        rewrites = all_queries[1:]

        embed_resp = client.embeddings.create(model=EMBED_MODEL, input=all_queries)
        q_vecs = [item.embedding for item in embed_resp.data]

        best: dict[str, dict] = {}
        for q_vec in q_vecs:
            results = collection.query(query_embeddings=[q_vec], n_results=min(TOP_K, collection.count()))
            for doc, meta, dist, cid in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                score = 1.0 - dist
                if cid not in best or score > best[cid]["score"]:
                    best[cid] = {"doc": doc, "meta": meta, "score": score}

        top_chunks = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:TOP_DEDUP]

        context_parts = []
        for i, chunk in enumerate(top_chunks, start=1):
            context_parts.append(
                f"[Chunk {i} | Section: {chunk['meta']['section_heading']} "
                f"| Source: {chunk['meta']['source_url']}]\n{chunk['doc']}"
            )
        context = "\n\n".join(context_parts)

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()

        print(f"--- QUERY {query_num} ---")
        print(f"Original: {query}")
        print("\nRewrites:")
        for i, rw in enumerate(rewrites, start=1):
            print(f"  {i}. {rw}")
        print("\nRetrieved Chunks (deduplicated across all 4 variants):")
        for i, chunk in enumerate(top_chunks, start=1):
            print(f"\n  Chunk {i} (score: {chunk['score']:.2f}): {chunk['meta']['section_heading']}")
            print(f"  {chunk['doc'][:300]}{'...' if len(chunk['doc']) > 300 else ''}")
        print("\nLLM Response:")
        print(answer)
        print()


if __name__ == "__main__":
    main()
