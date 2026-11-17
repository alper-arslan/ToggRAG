# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG system that answers how-to questions from a car owner's manual (PDF), with page citations. The manual is only a use case; the goal is to show RAG engineering, so prefer general, config-driven code over manual-specific rules.

Pipeline: parse PDF -> clean -> chunk -> embed -> Qdrant + BM25 -> hybrid retrieval (RRF) -> optional rerank -> LLM answer. No LangChain/LlamaIndex; embeddings, reranker and LLM sit behind small interfaces so providers can be swapped via config.

## Layout

- `src/toggrag/` (flow of a question: ingest -> retrieve -> answer):
  - Foundations: `config.py` (pydantic settings), `models.py` (Chunk, Page, Heading, Citation, Answer), `shared.py` (StrictModel base), `text.py` (Turkish lowercase, tokenize, stem, page cleaning).
  - Ingest: `pdf.py` (pages + bookmarks), `chunking.py` (sections -> chunks), `embeddings.py` (Embedder), `store.py` (Qdrant), `pipeline.py` (ingest, chunk persistence, RagPipeline).
  - Retrieve: `bm25.py`, `fusion.py` (RRF), `rerank.py` (optional Reranker), `retrieval.py` (HybridRetriever with bm25/dense/hybrid modes).
  - Answer: `llm.py` (LLM adapters), `answer.py` (grounded prompt, citation parsing).
  - Interfaces: `cli.py` (`toggrag ingest|ask|eval|ui`), `ui.py` (Gradio), `evaluation.py` (Hit@k, Recall@k, MRR, faithfulness).
- `config.yaml` holds tunables; secrets live only in `.env` (see `.env.example`).
- `eval/questions.yaml` is the gold set (question + answer pages).
- `data/` is gitignored; the manual goes in `data/manual.pdf`, the built index is written next to it.
- `Dockerfile` and `docker-compose.yml` serve the chat page; run `toggrag ingest` on the host first.

## Setup

- Python >=3.14, managed with uv. Use `uv sync`, `uv add <pkg>` (`--dev` for tooling) and `uv run <cmd>`; don't use pip directly.

## Commands

- Test: `uv run pytest`; single test: `uv run pytest path/to/test_file.py::test_name`
- Lint: `uv run ruff check --fix`
- Format: `uv run ruff format`

## Testing

- Test our own logic (validators, cleaning, chunking, fusion, metrics), not pydantic or PyMuPDF behaviour.
- Tests build plain objects (`Page`, `Heading`, `Chunk`) and do not need the real PDF.
- CI runs `ruff check`, `ruff format --check` and `pytest`; run them before committing.
