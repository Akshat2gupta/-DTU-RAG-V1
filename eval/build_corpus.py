#!/usr/bin/env python3
"""
Build the full chunk corpus from raw files (no Qdrant / OpenAI needed).

Reads every manifest row that has a file_path, runs the appropriate parser,
and writes all chunks to eval/all_chunks.jsonl.  Also prints per-source stats.

Run from repo root:
    python eval/build_corpus.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_ROOT      = _HERE.parent
_CHATBOT   = _ROOT / "dtu-chatbot"

for _p in (str(_CHATBOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ingestion.faculty_parser import parse_faculty_html
from ingestion.html_parser    import parse_html_file
from ingestion.ir_chunker     import chunk_document
from ingestion.pdf_parser     import parse_pdf

MANIFEST_DB = _CHATBOT / "manifest" / "manifest.db"
OUT_PATH    = _HERE / "all_chunks.jsonl"

_DEPT_RE    = re.compile(r"Departments/([^/]+)/faculty", re.IGNORECASE)


HTML_CATEGORIES = {
    "ordinance_html", "notice_html", "scholarship_html",
    "academics_php", "programme_html", "pg_academics_php",
    "dept_about", "dept_faculty",
    "about_php", "admin_php",
    "rnd_html", "nceet_html", "icc_html", "enggcell_html", "vigilance_html",
    "governance_php", "nirf_html",
    "admissions_html", "saarthi_html",
    "exam_html", "hostel_html", "tnp_html", "library_html",
}

PDF_CATEGORIES = {
    "ordinance_pdf", "scholarship_pdf", "admissions_pdf",
    "tnp_pdf", "notice_pdf",
}


def _chunks_for_row(row: dict) -> list[dict]:
    fp_raw = row.get("file_path") or ""
    if not fp_raw:
        return []
    file_path = Path(fp_raw) if Path(fp_raw).is_absolute() else _CHATBOT / fp_raw
    if not file_path.exists():
        return []

    cat = row.get("category", "")
    url = row["url"]

    try:
        if cat == "dept_faculty":
            m    = _DEPT_RE.search(url)
            dept = m.group(1) if m else None
            doc  = parse_faculty_html(file_path, url=url, dept=dept)
        elif cat in HTML_CATEGORIES:
            doc = parse_html_file(file_path, url=url, doc_type=cat)
        elif cat in PDF_CATEGORIES:
            title    = row.get("title") or file_path.stem.replace("_", " ")
            date_pub = row.get("date_published") or ""
            doc = parse_pdf(file_path, url=url, doc_type=row.get("document_type") or "pdf",
                            title=title, date_published=date_pub)
        else:
            return []

        return chunk_document(doc)

    except Exception as exc:
        print(f"  ERROR {url}: {exc}", flush=True)
        return []


def main() -> None:
    with sqlite3.connect(MANIFEST_DB) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM documents WHERE file_path IS NOT NULL ORDER BY category, id"
        )]

    print(f"Manifest rows with file_path: {len(rows)}")

    stats: dict[str, list[int]] = defaultdict(list)
    all_chunks: list[dict] = []
    seen_texts: set[str] = set()
    dupes = 0

    for i, row in enumerate(rows, 1):
        cat   = row.get("category", "?")
        label = row["url"].replace("https://", "").replace("http://", "")[:70]
        print(f"[{i:3d}/{len(rows)}] {cat:<22} {label}", flush=True)

        chunks = _chunks_for_row(row)
        n = 0
        for c in chunks:
            h = c["text"]
            if h in seen_texts:
                dupes += 1
                continue
            seen_texts.add(h)
            all_chunks.append(c)
            n += 1
            stats[cat].append(c["token_count"])

        print(f"          -> {n} chunks", flush=True)

    # Write output
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Total unique chunks : {len(all_chunks)}")
    print(f"Cross-source dupes  : {dupes}")
    print(f"\nPer-category stats:")
    for cat, toks in sorted(stats.items()):
        if not toks:
            continue
        print(f"  {cat:<25} n={len(toks):<6} tok_avg={sum(toks)//len(toks):<5} tok_max={max(toks)}")
    print(f"\nOutput: {OUT_PATH}")


if __name__ == "__main__":
    main()
