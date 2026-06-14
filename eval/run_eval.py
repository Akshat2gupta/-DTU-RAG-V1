#!/usr/bin/env python3
"""
Retrieval + answer eval for the DTU RAG pipeline.

For every item in eval_set.json:
  1. Retrieve top-5 chunks through the same path the app uses
     (multi-variant query rewrite + dedup), or plain single-vector
     retrieval with --no-rewrite.
  2. A retrieved chunk is "correct" when its text contains at least one
     of expect_keywords_any AND its source_url contains
     expect_source_contains (empty fragment = any source).
  3. Metrics: Hit@1, Hit@3, Hit@5, MRR over answerable items.
  4. With --answers: also generate the final LLM answer and check that
     unanswerable items trigger the no-answer guard.

Run from repo root:
    python eval/run_eval.py                 # retrieval only, with rewrites
    python eval/run_eval.py --no-rewrite    # cheaper, deterministic
    python eval/run_eval.py --answers       # + generation check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_ROOT / "vertical_slice"), str(_ROOT / "dtu-chatbot"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv(_ROOT / "vertical_slice" / ".env")

import os
from openai import OpenAI
from qdrant_client import QdrantClient

from e2e_pipeline import (
    CHAT_MODEL,
    COLLECTION_NAME,
    SYSTEM_PROMPT,
    _embed_batch,
    _expand_table_chunks,
    _rewrite_query,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOP_K = 5

_NO_ANSWER_PHRASES = (
    "i could not find this information",
    "could not find",
    "not available in",
    "no information",
    "not found in",
    "does not contain",
)


def retrieve(oai: OpenAI, qdrant: QdrantClient, query: str, rewrite: bool) -> list[dict]:
    variants = _rewrite_query(oai, query) if rewrite else [query]
    vectors = _embed_batch(oai, variants)
    best: dict[str, dict] = {}
    for vec in vectors:
        hits = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=vec,
            limit=TOP_K, with_payload=True,
        ).points
        for h in hits:
            cid = h.payload["chunk_id"]
            if cid not in best or h.score > best[cid]["score"]:
                best[cid] = {"payload": h.payload, "score": h.score}
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:TOP_K]


def chunk_is_correct(payload: dict, item: dict) -> bool:
    text = payload.get("text", "").lower()
    url = (payload.get("source_url") or "").lower()
    kws = [k.lower() for k in item.get("expect_keywords_any", [])]
    frag = (item.get("expect_source_contains") or "").lower()
    kw_ok = any(k in text for k in kws) if kws else True
    src_ok = frag in url if frag else True
    return kw_ok and src_ok


def generate_answer(oai: OpenAI, qdrant: QdrantClient, query: str, top: list[dict]) -> str:
    # Same small-to-big table expansion the app uses; retrieval metrics above
    # are computed on the raw chunks, generation sees the full tables.
    top = _expand_table_chunks(qdrant, top)
    context = "\n\n".join(
        f"[Chunk {i} | Section: {c['payload'].get('section_heading','')} "
        f"| Source: {c['payload'].get('source_url','')}]\n{c['payload']['text']}"
        for i, c in enumerate(top, 1)
    )
    resp = oai.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--answers", action="store_true",
                    help="Also generate LLM answers and check the no-answer guard")
    ap.add_argument("--only", help="Run a single item id")
    ap.add_argument("--eval-set", default="eval_set.json",
                    help="Eval set filename inside eval/ (or an absolute path)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    eval_path = Path(args.eval_set)
    if not eval_path.is_absolute():
        eval_path = _HERE / eval_path
    items = json.loads(eval_path.read_text(encoding="utf-8"))["items"]
    if args.only:
        items = [i for i in items if i["id"] == args.only]

    oai = OpenAI()
    qdrant = QdrantClient(host="localhost", port=6333)

    results = []
    hits1 = hits3 = hits5 = 0
    mrr_total = 0.0
    n_answerable = 0
    guard_pass = guard_total = 0

    def _with_retry(fn, *fn_args):
        """Run fn, recreating the OpenAI client on failure.

        api.openai.com intermittently returns 431 (request_headers_too_large)
        when the client's httpx session has accumulated Cloudflare cookies
        over many calls; a fresh client clears them."""
        nonlocal oai
        last_exc = None
        for attempt in range(3):
            try:
                return fn(oai, *fn_args)
            except Exception as exc:
                last_exc = exc
                print(f"    retry {attempt + 1}/3 after error: {exc}")
                time.sleep(2)
                oai = OpenAI()
        raise last_exc

    t0 = time.perf_counter()
    for item in items:
        top = _with_retry(
            lambda c: retrieve(c, qdrant, item["question"], rewrite=not args.no_rewrite)
        )
        first_correct = next(
            (i for i, c in enumerate(top, 1) if chunk_is_correct(c["payload"], item)),
            None,
        )

        rec = {
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "first_correct_rank": first_correct,
            "top_score": top[0]["score"] if top else 0,
            "retrieved": [
                {
                    "rank": i,
                    "score": round(c["score"], 3),
                    "title": c["payload"].get("document_title", ""),
                    "section": c["payload"].get("section_heading", "")[:60],
                    "url": c["payload"].get("source_url", ""),
                    "correct": chunk_is_correct(c["payload"], item),
                    "snippet": c["payload"].get("text", "")[:150],
                }
                for i, c in enumerate(top, 1)
            ],
        }

        if item["answerable"]:
            n_answerable += 1
            if first_correct:
                mrr_total += 1.0 / first_correct
                hits1 += first_correct <= 1
                hits3 += first_correct <= 3
                hits5 += first_correct <= 5
            status = f"hit@{first_correct}" if first_correct else "MISS"
        else:
            status = "n/a (unanswerable)"

        if args.answers:
            ans = _with_retry(
                lambda c: generate_answer(c, qdrant, item["question"], top)
            )
            rec["answer"] = ans
            if not item["answerable"]:
                guard_total += 1
                refused = any(p in ans.lower() for p in _NO_ANSWER_PHRASES)
                rec["guard_refused"] = refused
                guard_pass += refused
                status = "GUARD-OK" if refused else "GUARD-FAIL (hallucinated)"

        print(f"[{item['id']:9s}] {status:12s} top={rec['top_score']:.3f}  {item['question'][:70]}")
        if item["answerable"] and not first_correct:
            for r in rec["retrieved"][:3]:
                print(f"      miss-> [{r['score']}] {r['title'][:40]} | {r['section']}")
        results.append(rec)

    print("\n" + "=" * 60)
    if n_answerable:
        print(f"Answerable items : {n_answerable}")
        print(f"Hit@1 : {hits1}/{n_answerable}  ({hits1/n_answerable:.0%})")
        print(f"Hit@3 : {hits3}/{n_answerable}  ({hits3/n_answerable:.0%})")
        print(f"Hit@5 : {hits5}/{n_answerable}  ({hits5/n_answerable:.0%})")
        print(f"MRR   : {mrr_total/n_answerable:.3f}")
    if guard_total:
        print(f"No-answer guard : {guard_pass}/{guard_total} refused correctly")
    print(f"Time  : {time.perf_counter() - t0:.0f}s")

    out = _HERE / (eval_path.stem.replace("eval_set", "eval_results") + ".json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detail: {out}")


if __name__ == "__main__":
    main()
