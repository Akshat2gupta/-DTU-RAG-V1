# Vertical Slice — Prototype

This is a **prototype**, not production code.

It was built to validate the end-to-end RAG pipeline early: embed chunks, index them, run queries, and inspect results. The goal was to confirm the approach before investing in a full implementation.

## What's here

- `test_rag.py` — embed + FAISS index + query loop; accepts any JSONL chunks file
- `multi_agent_board.py` — early multi-agent orchestration experiment
- `chunks.json` — hand-curated sample chunks used for initial testing
- `requirements.txt` — dependencies for this slice only

## Running it

```bash
cd vertical_slice
pip install -r requirements.txt

# Against the hand-curated chunks
python test_rag.py

# Against a chunks file from the pipeline
python test_rag.py --chunks ../dtu-chatbot/data/chunks/BTech_2022_ordinance_ir_chunks.jsonl \
    --query "What is the minimum attendance required?"
```

## Status

Not maintained. The production pipeline lives in `dtu-chatbot/`.
