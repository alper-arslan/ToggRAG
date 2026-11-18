# ToggRag

A RAG system that answers how-to questions from a car owner's manual (a Turkish
Togg T10X manual PDF) and cites the manual pages it used. The manual is only a
use case: the point is the pipeline, so everything is config-driven and the
answers are measured, not just demoed.

No LangChain or LlamaIndex. Embeddings, reranker and LLM sit behind small
interfaces (`Embedder`, `Reranker`, `LLM`) and are chosen in `config.yaml`.

```
PDF -> pages + bookmarks -> clean -> section-aware chunks
    -> embeddings (Qdrant)  \
                             -> RRF fusion -> [cross-encoder rerank] -> grounded LLM answer + page citations
    -> BM25 (Turkish stems) /
```

## Quick start

Needs Python 3.14 and [uv](https://docs.astral.sh/uv/). The manual is not in the
repo (it is a public download from the Togg site); put it at `data/manual.pdf`.

```bash
uv sync                       # add --extra ui for the chat page, --extra rerank for the reranker
ollama pull bge-m3 && ollama pull qwen2.5:7b    # default local models (any other provider works)
uv run toggrag ingest         # parse, chunk, embed, index
uv run toggrag ask "Şarj kapağı açılmazsa ne yapmalıyım?"
uv run toggrag ui             # minimal chat page on http://127.0.0.1:7860
uv run toggrag eval           # compare BM25 / dense / hybrid / hybrid+rerank
```

Docker: `docker compose up --build` serves the chat page and talks to Ollama on
the host. Mount your own `data/` (it holds the PDF and the built index).

Use Claude or OpenAI instead of a local model: set `llm.provider` and
`llm.model` in `config.yaml` and put the key in `.env` (see `.env.example`).
Any setting can also be overridden by an environment variable, e.g.
`LLM__MODEL=qwen2.5:14b`.

## Design decisions

- **Structure-aware chunking.** The PDF's bookmarks give the heading hierarchy,
  so sections are cut at real headings instead of every N characters. Tiny
  sections of one chapter are merged up to `chunking.target_tokens`; big ones
  are split only between blocks (list items, callouts, sentences), never inside
  one. Adjacent chunks overlap by `overlap_ratio`. Each chunk stores its pages,
  section path and a content type (procedure, warning, specification, text).
- **Contextual headers.** The section path is put in front of the chunk text
  before embedding and BM25 indexing, so "3.4.1 Kaputun Açılması" is part of
  what the chunk is about.
- **Hybrid retrieval with RRF.** BM25 catches exact terms and part names, dense
  search catches paraphrases. Reciprocal Rank Fusion merges the two by rank, so
  no score normalisation is needed.
- **Turkish text handling.** Correct dotted/dotless I lowercasing and a Snowball
  stemmer for BM25. Known limit: it strips noun suffixes (`ayarlar -> ayar`) but
  leaves most verb forms alone (`ayarlanır`).
- **Grounded answers.** The prompt allows only the numbered excerpts, requires
  `[n]` citations and a fixed refusal sentence when the manual has no answer.
  Citations are parsed from the answer and checked against the retrieved
  chunks; if retrieval finds nothing the LLM is not called at all (token saving).
- **Local first, model agnostic.** Default is Ollama through its native chat
  API; other adapters speak the OpenAI chat format (OpenAI, vLLM...) or the
  Anthropic Messages API. Ollama's default context window is 2048 tokens and it
  silently cuts the start of a longer prompt, so `llm.context_window` (default
  8192) is sent with every request, and a warning is logged if a prompt still
  fills it. Five chunks plus the rules need roughly 5k tokens. For Ollama
  `llm.base_url` is the server root; a trailing `/v1` is dropped.
- **Qdrant in local mode.** No server to run; the same code works against a
  Qdrant server later.

## Evaluation

`eval/questions.yaml` holds gold questions with the manual pages that answer
them. A retrieved chunk is relevant if it covers one of those pages. `toggrag
eval` prints Hit@k, Recall@k and MRR for BM25, dense, hybrid and hybrid+rerank;
`--faithfulness` also asks the LLM to judge whether each answer is fully
supported by its excerpts.

Measured so far (13 questions, BM25 only, no model needed):

| config | Hit@1 | Hit@3 | Hit@5 | MRR |
|--------|-------|-------|-------|-----|
| bm25   | 0.92  | 1.00  | 1.00  | 0.95 |

Dense, hybrid and rerank rows need an embedding model; run `toggrag eval` to
fill them in on your machine. The gold set is small and its questions share many
words with the headings, so it favours BM25: treat it as a smoke test and grow
it with harder, paraphrased questions before drawing conclusions.

## Limitations

- Page cleaning is a heuristic for this manual (figure ids, callout numbers).
  Another manual may need its own rules.
- Chunks merged from several subsections only keep their shared section path.
- Tables are not parsed as tables yet.

## Development

```bash
uv run ruff check --fix && uv run ruff format && uv run pytest
```
