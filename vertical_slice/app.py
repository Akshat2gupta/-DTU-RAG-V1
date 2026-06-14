#!/usr/bin/env python3
"""
DTU RAG -- Streamlit chat UI.

Run from the repo root (Rag_V1/):
    streamlit run vertical_slice/app.py

Requires Qdrant running and the collection already indexed:
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
        -v qdrant_storage:/qdrant/storage qdrant/qdrant
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_CHATBOT   = _REPO_ROOT / "dtu-chatbot"

for _p in [str(_CHATBOT), str(_REPO_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from e2e_pipeline import (
    CHAT_MODEL,
    COLLECTION_NAME,
    SYSTEM_PROMPT,
    TOP_DEDUP,
    TOP_K,
    _embed_batch,
    _expand_table_chunks,
    _rewrite_query,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DTU Assistant",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cached clients
# ---------------------------------------------------------------------------


@st.cache_resource
def get_clients() -> tuple[OpenAI, QdrantClient]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set in vertical_slice/.env")
        st.stop()
    return (
        OpenAI(api_key=api_key),
        QdrantClient(host="localhost", port=6333),
    )


oai, qdrant = get_clients()

# ---------------------------------------------------------------------------
# Check collection exists
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def _collection_point_count() -> int:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION_NAME not in existing:
        return 0
    return qdrant.get_collection(COLLECTION_NAME).points_count or 0


n_docs = _collection_point_count()
if n_docs == 0:
    st.error(
        f"Qdrant collection '{COLLECTION_NAME}' is empty or missing. "
        "Run e2e_pipeline.py to index documents first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("DTU Assistant")
st.sidebar.success(f"**{COLLECTION_NAME}**  \n{n_docs:,} chunks indexed")

_YEAR_OPTIONS = ["All years (no filter)", 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]
_selected_year_label = st.sidebar.selectbox(
    "Ordinance batch year",
    options=_YEAR_OPTIONS,
    index=0,
    help="Restrict answers to a specific B.Tech ordinance year. "
         "'All years' always includes evergreen content (hostel, dept pages, etc.).",
)
_batch_year_filter: int | None = (
    None if _selected_year_label == "All years (no filter)" else int(_selected_year_label)
)

st.sidebar.divider()
st.sidebar.caption(
    "Answers are based solely on indexed DTU documents. "
    "Always verify with the official DTU source."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Retrieval (separate from generation so spinner fires before streaming)
# ---------------------------------------------------------------------------

_LOW_SCORE_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Curated documents (hand-verified — sources that machine extraction mangles,
# like scanned T&P stats sheets and cutoff tables). Queries matching a doc's
# intent always get the full doc in the LLM context AND a visible expander
# the user can read directly.
# ---------------------------------------------------------------------------

_CURATED_DIR  = _CHATBOT / "data" / "curated"
_CURATED_DOCS = [
    {
        "key":   "placement",
        "path":  _CURATED_DIR / "placement_stats.md",
        "intent": re.compile(
            r"placement|package|\bctc\b|\blpa\b|placed|salary|recruit",
            re.IGNORECASE),
        "title": "📊 Full DTU placement statistics — all branches, 2019-2025",
        "label": "DTU Placement Statistics (hand-verified, all branches 2019-2025)",
    },
    {
        "key":   "admissions",
        "path":  _CURATED_DIR / "admissions.md",
        "intent": re.compile(
            r"cutoff|cut-off|closing rank|admission|counsell?ing|\bjac\b|\bjee\b"
            r"|eligib|seat matrix|seats|registration|last date.*apply"
            r"|\bfees?\b|tuition",
            re.IGNORECASE),
        "title": "🎓 DTU admissions — cutoffs 2025, JAC 2026 dates, eligibility, seats, fees",
        "label": "DTU Admissions (hand-verified: JEE cutoffs 2025, JAC Delhi 2026 "
                 "schedule, eligibility, seat matrix, fees)",
    },
    {
        "key":   "academic_rules",
        "path":  _CURATED_DIR / "academic_rules.md",
        "intent": re.compile(
            r"backlog|\bfail|\bf grade\b|'f' grade|supplementary|reappear"
            r"|re-appear|repeat (a |the )?course|summer semester|detain"
            r"|\bgrade\b|grading|sgpa|cgpa|attendance|passing|re-?evaluat"
            r"|recheck|marks|percentage|credits",
            re.IGNORECASE),
        "title": "📚 DTU academic rules — grading, attendance, backlogs, exams",
        "label": "DTU Academic Rules (hand-verified from B.Tech ordinances: "
                 "grading table, SGPA/CGPA, attendance, backlogs, summer "
                 "semester, re-evaluation)",
    },
    {
        "key":   "hostel",
        "path":  _CURATED_DIR / "hostel.md",
        "intent": re.compile(
            r"hostel|mess\b|warden|curfew|accommodation|room rent",
            re.IGNORECASE),
        "title": "🏠 DTU hostels — fees, rules, allotment, mess",
        "label": "DTU Hostels (hand-verified from Hostel Bulletin 2024-25: "
                 "fees, allotment, rules, mess)",
    },
]


@st.cache_data
def _curated_md_cached(path_str: str, mtime: float) -> str:
    """Cached by (path, mtime) — editing a curated doc on disk invalidates
    the cache automatically; no app restart needed."""
    return Path(path_str).read_text(encoding="utf-8")


def _curated_md(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    return _curated_md_cached(path_str, p.stat().st_mtime)


def _matched_curated(q: str) -> list[dict]:
    return [d for d in _CURATED_DOCS if d["intent"].search(q) and _curated_md(str(d["path"]))]


def _render_curated(keys: list[str]) -> None:
    for doc in _CURATED_DOCS:
        if doc["key"] in keys:
            with st.expander(doc["title"], expanded=False):
                st.markdown(_curated_md(str(doc["path"])))


def _retrieve(query: str, batch_year: int | None) -> tuple[str, list[dict], float]:
    """Rewrite → embed → retrieve. Returns (context_str, sources, top_score)."""
    variants  = _rewrite_query(oai, query)
    q_vectors = _embed_batch(oai, variants)

    qfilter = None
    if batch_year:
        qfilter = Filter(
            should=[
                FieldCondition(key="batch_year", match=MatchValue(value=batch_year)),
                FieldCondition(key="batch_year", match=MatchValue(value=0)),
            ]
        )

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
            score = hit.score
            if cid not in best or score > best[cid]["score"]:
                best[cid] = {"payload": hit.payload, "score": score}

    top        = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:TOP_DEDUP]
    top        = _expand_table_chunks(qdrant, top)
    top_score  = top[0]["score"] if top else 0.0

    context_parts = [
        f"[Chunk {i} | Section: {c['payload']['section_heading']} "
        f"| Source: {c['payload']['source_url']}]\n{c['payload']['text']}"
        for i, c in enumerate(top, 1)
    ]
    context = "\n\n".join(context_parts)

    sources = [
        {
            "rank":       i,
            "score":      c["score"],
            "page":       c["payload"].get("page_number", "?"),
            "section":    c["payload"].get("section_heading", ""),
            "block_type": c["payload"].get("block_type", "?"),
            "source_url": c["payload"].get("source_url", ""),
            "preview":    c["payload"].get("text", "")[:250] + "…",
        }
        for i, c in enumerate(top, 1)
    ]
    return context, sources, top_score


_NO_ANSWER_PHRASES = (
    "i could not find this information",
    "don't have information",
    "not mentioned in",
    "not available in",
    "cannot find",
    "no information",
    "not covered",
    "not found in",
    "does not contain",
    "i cannot answer",
    "outside the scope",
    "not in the provided",
)

_REDIRECT_RULES: list[tuple[tuple[str, ...], str, str]] = [
    # (keywords, label, url)
    (
        ("admission", "jee", "counsell", "seat allot", "cutoff", "merit list"),
        "DTU Admissions Portal (JAC Delhi)",
        "https://jacdelhi.admissions.nic.in",
    ),
    (
        ("placement", "tnp", "tpc", "recruit", "intern", "package", "lpa", "company", "campus"),
        "DTU Training & Placement Cell",
        "https://tnp.dtu.ac.in/index.html",
    ),
    (
        ("fee", "erp", "timetable", "time table", "registration", "enroll"),
        "DTU Student Portal",
        "https://dtu.ac.in",
    ),
    (
        ("faculty", "professor", "mentor", "guide", "supervisor", "who should", "which prof",
         "kavinder", "kavindra"),
        "DTU Faculty Directory",
        "https://dtu.ac.in/Web/Departments/",
    ),
]


def _is_no_answer(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _NO_ANSWER_PHRASES)


def _redirect_suggestion(query: str) -> tuple[str, str] | None:
    """Return (label, url) for the most relevant external resource, or None."""
    q = query.lower()
    for keywords, label, url in _REDIRECT_RULES:
        if any(kw in q for kw in keywords):
            return label, url
    return None


def _openai_stream(stream):
    """Yield text deltas from an OpenAI streaming response."""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Source expander (shared by history render + new messages)
# ---------------------------------------------------------------------------


def _render_sources(sources: list[dict]) -> None:
    with st.expander("Sources", expanded=False):
        for src in sources:
            url  = src.get("source_url", "")
            link = f"[{url}]({url})" if url.startswith("http") else f"`{url}`"
            st.markdown(
                f"**[{src['rank']}]** score `{src['score']:.3f}` &nbsp;|&nbsp; "
                f"page `{src['page']}` &nbsp;|&nbsp; `{src['block_type']}`  \n"
                f"*{src['section'][:80]}*  \n"
                f"{link}"
            )
            st.caption(src["preview"])
            st.divider()


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("DTU Institutional Assistant")
st.caption("Ask about attendance, grading, credits, fees, hostel, faculty, and more.")

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _hist_keys = msg.get("curated_keys") or (["placement"] if msg.get("placement_table") else [])
        if _hist_keys:
            _render_curated(_hist_keys)
        if msg.get("sources"):
            _render_sources(msg["sources"])

# ---------------------------------------------------------------------------
# Starter questions (only shown on an empty chat)
# ---------------------------------------------------------------------------

_STARTER_QUESTIONS = [
    "What is the minimum attendance requirement?",
    "How is SGPA calculated?",
    "What are the grading criteria for B.Tech?",
    "How many credits are required to graduate?",
    "What happens if a student fails a subject?",
    "What is the fee structure for B.Tech?",
]

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(_STARTER_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True, key=f"starter_{i}"):
            st.session_state["_pending_query"] = q
            st.rerun()

# ---------------------------------------------------------------------------
# Chat input + query handler
# ---------------------------------------------------------------------------

_typed_query = st.chat_input("Ask about DTU regulations, attendance, grades, fees…")
_pending     = st.session_state.pop("_pending_query", None)
query        = _pending or _typed_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Phase 1: retrieve
        with st.spinner("Searching DTU documents…"):
            context, sources, top_score = _retrieve(query, batch_year=_batch_year_filter)

        # Queries matching a curated doc's intent always get the full
        # hand-verified doc in the context — retrieval alone can surface
        # garbled machine-extracted numbers or nothing at all.
        _matched = _matched_curated(query)
        for _doc in _matched:
            context += (
                f"\n\n[Curated chunk | Section: {_doc['label']}]\n"
                + _curated_md(str(_doc["path"]))
            )

        # Phase 2: low-confidence warning before we even start generating
        if top_score < _LOW_SCORE_THRESHOLD:
            st.warning(
                "No closely matching documents found — the answer below may not be accurate. "
                "Please verify with the official DTU website.",
                icon="⚠️",
            )

        # Phase 3: stream generation
        stream = oai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        answer = st.write_stream(_openai_stream(stream))

        # Phase 4: "no answer" info box (shown after full text is streamed)
        if _is_no_answer(answer):
            redirect = _redirect_suggestion(query)
            if redirect:
                label, url = redirect
                st.info(
                    f"This topic isn't in the indexed DTU documents. "
                    f"Try **[{label}]({url})**.",
                    icon="ℹ️",
                )
            else:
                st.info(
                    "This topic isn't in the indexed DTU documents. "
                    "Try rephrasing, or visit **[dtu.ac.in](https://dtu.ac.in)**.",
                    icon="ℹ️",
                )

        # Phase 5: full curated docs — always visible for matching queries so
        # the user can read the official numbers directly even when the
        # generated answer is incomplete.
        _matched_keys = [d["key"] for d in _matched]
        if _matched_keys:
            _render_curated(_matched_keys)

        # Phase 6: sources
        if sources:
            _render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "curated_keys": _matched_keys,
        }
    )
