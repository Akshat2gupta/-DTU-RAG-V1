#!/usr/bin/env python3
"""
DTU RAG -- Streamlit chat UI.

Run from the repo root (Rag_V1/):
    streamlit run vertical_slice/app.py

Requires Qdrant to be running and the collection already indexed:
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
        -v qdrant_storage:/qdrant/storage qdrant/qdrant
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# -- path bootstrap (same as e2e_pipeline.py) --------------------------------
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

from e2e_pipeline import (
    CHAT_MODEL,
    COLLECTION_NAME,
    SYSTEM_PROMPT,
    TOP_DEDUP,
    TOP_K,
    _embed_batch,
    _rewrite_query,
)
from qdrant_client.models import FieldCondition, Filter, MatchValue

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DTU Assistant",
    page_icon="🎓",
    layout="centered",
)

st.title("DTU Institutional Assistant")
st.caption(
    "Answers are based solely on indexed DTU documents (B.Tech Ordinance 2022). "
    "Always verify with the official source."
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

st.sidebar.success(f"Collection: **{COLLECTION_NAME}**  \n{n_docs} chunks indexed")

# ---------------------------------------------------------------------------
# Batch year filter
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "sources": ...}

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**[{src['rank']}]** score `{src['score']:.3f}` &nbsp;|&nbsp; "
                        f"page `{src['page']}` &nbsp;|&nbsp; `{src['block_type']}`  \n"
                        f"*{src['section'][:80]}*"
                    )
                    st.caption(src["preview"])
                    st.divider()

# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------


def _run_query(query: str, batch_year: int | None = None) -> tuple[str, list[dict]]:
    """Rewrite → embed → retrieve → synthesize. Returns (answer, sources)."""
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

    top = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:TOP_DEDUP]

    context_parts = [
        f"[Chunk {i} | Section: {c['payload']['section_heading']} "
        f"| Source: {c['payload']['source_url']}]\n{c['payload']['text']}"
        for i, c in enumerate(top, 1)
    ]
    context = "\n\n".join(context_parts)

    resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    answer = resp.choices[0].message.content.strip()

    sources = [
        {
            "rank":       i,
            "score":      c["score"],
            "page":       c["payload"].get("page_number", "?"),
            "section":    c["payload"].get("section_heading", ""),
            "block_type": c["payload"].get("block_type", "?"),
            "preview":    c["payload"].get("text", "")[:250] + "…",
        }
        for i, c in enumerate(top, 1)
    ]
    return answer, sources


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if query := st.chat_input("Ask about DTU regulations, attendance, grades, fees…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run query and stream answer
    with st.chat_message("assistant"):
        with st.spinner("Searching DTU documents…"):
            answer, sources = _run_query(query, batch_year=_batch_year_filter)

        st.markdown(answer)

        if sources:
            with st.expander("Sources", expanded=False):
                for src in sources:
                    st.markdown(
                        f"**[{src['rank']}]** score `{src['score']:.3f}` &nbsp;|&nbsp; "
                        f"page `{src['page']}` &nbsp;|&nbsp; `{src['block_type']}`  \n"
                        f"*{src['section'][:80]}*"
                    )
                    st.caption(src["preview"])
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
