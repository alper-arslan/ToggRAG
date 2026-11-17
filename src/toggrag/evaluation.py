from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import Field, PositiveInt

from toggrag.answer import answer_question, format_context
from toggrag.config import Settings, load_settings
from toggrag.embeddings import make_embedder
from toggrag.llm import LLM, make_llm
from toggrag.models import Chunk
from toggrag.pipeline import load_retriever
from toggrag.shared import StrictModel

KS = (1, 3, 5)

JUDGE_PROMPT = """Aşağıda kılavuz alıntıları, bir soru ve bu soruya verilen bir yanıt var. \
Yanıttaki her bilgi alıntılarla destekleniyorsa yalnızca SUPPORTED, desteklenmeyen \
herhangi bir bilgi varsa yalnızca UNSUPPORTED yaz."""


class Question(StrictModel):
    """One gold question with the pages that answer it."""

    question: str
    pages: list[PositiveInt] = Field(min_length=1)


def load_questions(path: Path) -> list[Question]:
    return [Question(**row) for row in yaml.safe_load(path.read_text(encoding="utf-8"))]


def _relevant(chunk: Chunk, expected: list[int]) -> bool:
    return bool(set(chunk.pages) & set(expected))


def hit_at_k(retrieved: list[Chunk], expected: list[int], k: int) -> float:
    """1.0 if any of the top k chunks has an expected page."""
    return float(any(_relevant(chunk, expected) for chunk in retrieved[:k]))


def recall_at_k(retrieved: list[Chunk], expected: list[int], k: int) -> float:
    """Share of the expected pages covered by the top k chunks."""
    found = {page for chunk in retrieved[:k] for page in chunk.pages}
    return len(found & set(expected)) / len(set(expected))


def reciprocal_rank(retrieved: list[Chunk], expected: list[int]) -> float:
    """1 / rank of the first relevant chunk; 0 when there is none."""
    for rank, chunk in enumerate(retrieved, start=1):
        if _relevant(chunk, expected):
            return 1 / rank
    return 0.0


def evaluate(
    retrieve: Callable[[str], list[Chunk]], questions: list[Question]
) -> dict[str, float]:
    """Average Hit@k, Recall@k and MRR of a retrieval function over the questions."""
    results = [(q, retrieve(q.question)) for q in questions]
    metrics: dict[str, float] = {}
    for k in KS:
        metrics[f"Hit@{k}"] = sum(hit_at_k(r, q.pages, k) for q, r in results)
        metrics[f"Recall@{k}"] = sum(recall_at_k(r, q.pages, k) for q, r in results)
    metrics["MRR"] = sum(reciprocal_rank(r, q.pages) for q, r in results)
    return {name: value / len(questions) for name, value in metrics.items()}


def judge_faithfulness(
    question: str, answer: str, chunks: list[Chunk], llm: LLM
) -> bool:
    """Ask an LLM whether every claim of the answer is supported by the chunks."""
    user = (
        f"Alıntılar:\n\n{format_context(chunks)}\n\nSoru: {question}\n\nYanıt: {answer}"
    )
    return llm.complete(JUDGE_PROMPT, user).strip().upper().startswith("SUPPORTED")


def format_table(rows: dict[str, dict[str, float]]) -> str:
    names = list(next(iter(rows.values())))
    lines = ["config".ljust(16) + "".join(name.rjust(10) for name in names)]
    for config, metrics in rows.items():
        lines.append(config.ljust(16) + "".join(f"{metrics[n]:10.2f}" for n in names))
    return "\n".join(lines)


def _with_reranker(settings: Settings, enabled: bool) -> Settings:
    reranker = settings.reranker.model_copy(update={"enabled": enabled})
    return settings.model_copy(update={"reranker": reranker})


def run_eval(questions_path: str, faithfulness: bool) -> None:
    """Compare BM25, dense, hybrid and hybrid + rerank on the gold questions."""
    settings = load_settings()
    questions = load_questions(Path(questions_path))
    embedder = make_embedder(settings.embedding)
    plain = load_retriever(_with_reranker(settings, False), embedder)
    rows = {
        "bm25": evaluate(lambda q: plain.retrieve(q, "bm25"), questions),
        "dense": evaluate(lambda q: plain.retrieve(q, "dense"), questions),
        "hybrid": evaluate(lambda q: plain.retrieve(q, "hybrid"), questions),
    }
    try:
        reranked = load_retriever(_with_reranker(settings, True), embedder)
        rows["hybrid+rerank"] = evaluate(reranked.retrieve, questions)
    except ImportError:
        print("skipping hybrid+rerank: run `uv sync --extra rerank` to enable it\n")
    print(format_table(rows))

    if faithfulness:
        llm = make_llm(settings)
        supported = 0
        for q in questions:
            chunks = plain.retrieve(q.question)
            answer = answer_question(q.question, chunks, llm)
            supported += judge_faithfulness(q.question, answer.text, chunks, llm)
        print(f"\nfaithfulness: {supported}/{len(questions)} answers fully supported")
