#!/usr/bin/env python3
"""
Offline BM25 retrieval eval — no Qdrant, no OpenAI needed.

Loads eval/all_chunks.jsonl (built by build_corpus.py) and evaluates
every query in eval_set.json using BM25 ranking.

BM25 is a strong lower-bound for semantic retrieval: if a chunk can't be
found by keyword overlap, it almost certainly won't be found by embedding
either (the reverse is NOT true).  Failing items here are guaranteed
retrieval gaps; passing items are not guaranteed to pass embedding eval.

Run:
    python eval/offline_eval.py
    python eval/offline_eval.py --verbose    # show top-5 per query
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[a-zA-Z0-9%@\.]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def chunk_is_correct(chunk: dict, item: dict) -> bool:
    text  = chunk.get("text", "").lower()
    url   = (chunk.get("source_url") or "").lower()
    kws   = [k.lower() for k in item.get("expect_keywords_any", [])]
    frag  = (item.get("expect_source_contains") or "").lower()
    kw_ok = any(k in text for k in kws) if kws else True
    src_ok = frag in url if frag else True
    return kw_ok and src_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="Print top-5 hits per query")
    ap.add_argument("--corpus", default=str(_HERE / "all_chunks.jsonl"), help="Chunk corpus JSONL")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}\nRun: python eval/build_corpus.py first")

    with open(corpus_path, encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f if l.strip()]

    items = json.loads((_HERE / "eval_set.json").read_text(encoding="utf-8"))["items"]

    print(f"Corpus : {len(chunks)} chunks")
    print(f"Queries: {len(items)} total  ({sum(1 for i in items if i['answerable'])} answerable)\n")

    # Build BM25 index
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    TOP_K = 5
    hits1 = hits3 = hits5 = 0
    mrr_total = 0.0
    n_answerable = 0

    results = []
    for item in items:
        if not item["answerable"]:
            print(f"[{item['id']:9s}] SKIP (unanswerable)")
            continue

        n_answerable += 1
        query_tokens = tokenize(item["question"])
        # also tokenize keyword hints to boost recall
        for kw in item.get("expect_keywords_any", []):
            query_tokens += tokenize(kw)

        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        top = [(idx, scores[idx], chunks[idx]) for idx in top_indices]

        first_correct = next(
            (rank for rank, (_, _, c) in enumerate(top, 1) if chunk_is_correct(c, item)),
            None,
        )

        if first_correct:
            mrr_total += 1.0 / first_correct
            hits1 += first_correct <= 1
            hits3 += first_correct <= 3
            hits5 += first_correct <= 5
            status = f"hit@{first_correct}"
        else:
            status = "MISS"

        print(f"[{item['id']:9s}] {status:8s}  {item['question'][:65]}")

        if args.verbose or not first_correct:
            for rank, (_, score, c) in enumerate(top, 1):
                correct = "✓" if chunk_is_correct(c, item) else " "
                url_frag = c.get("source_url", "")[-50:]
                print(f"  {correct} [{rank}] score={score:.1f}  {url_frag}  | {c['text'][:80]}")
            if not first_correct:
                # Show what a correct chunk looks like (first match in corpus)
                correct_chunks = [c for c in chunks if chunk_is_correct(c, item)]
                if correct_chunks:
                    c = correct_chunks[0]
                    print(f"  → CORRECT chunk exists: {c['source_url'][-50:]} | {c['text'][:100]}")
                else:
                    print(f"  → NO correct chunk found in corpus! Missing content.")

        results.append({"id": item["id"], "status": status, "first_correct_rank": first_correct})

    print(f"\n{'=' * 60}")
    print(f"Answerable items : {n_answerable}")
    if n_answerable:
        print(f"Hit@1 : {hits1}/{n_answerable}  ({hits1/n_answerable:.0%})")
        print(f"Hit@3 : {hits3}/{n_answerable}  ({hits3/n_answerable:.0%})")
        print(f"Hit@5 : {hits5}/{n_answerable}  ({hits5/n_answerable:.0%})")
        print(f"MRR   : {mrr_total/n_answerable:.3f}")

    out = _HERE / "offline_eval_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDetail: {out}")


if __name__ == "__main__":
    main()
