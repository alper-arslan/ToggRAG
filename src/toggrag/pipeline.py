from pathlib import Path

from toggrag.answer import answer_question
from toggrag.bm25 import BM25Index
from toggrag.chunking import chunk_document, contextual_text
from toggrag.config import Settings
from toggrag.embeddings import Embedder
from toggrag.llm import LLM
from toggrag.models import Answer, Chunk
from toggrag.pdf import read_headings, read_pages
from toggrag.rerank import make_reranker
from toggrag.retrieval import HybridRetriever, Mode
from toggrag.store import VectorStore
from toggrag.text import analyze


def chunks_path(settings: Settings) -> Path:
    return settings.data_dir / "chunks.jsonl"


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(chunk.model_dump_json() for chunk in chunks) + "\n", encoding="utf-8"
    )


def load_chunks(path: Path) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines if line]


def ingest(settings: Settings, embedder: Embedder) -> list[Chunk]:
    """Parse, chunk and embed the manual; write the chunks and the Qdrant collection."""
    chunks = chunk_document(
        read_pages(settings.pdf_path),
        read_headings(settings.pdf_path),
        settings.chunking,
    )
    vectors = embedder.embed([contextual_text(chunk) for chunk in chunks])
    VectorStore(settings.vector_store).rebuild(chunks, vectors)
    save_chunks(chunks, chunks_path(settings))
    return chunks


def load_retriever(
    settings: Settings, embedder: Embedder, store: VectorStore | None = None
) -> HybridRetriever:
    """Rebuild the BM25 index from the saved chunks and open the vector store."""
    chunks = load_chunks(chunks_path(settings))
    bm25 = BM25Index([analyze(contextual_text(chunk)) for chunk in chunks])
    return HybridRetriever(
        chunks,
        bm25,
        store or VectorStore(settings.vector_store),
        embedder,
        settings.retrieval,
        reranker=make_reranker(settings.reranker),
        reranker_config=settings.reranker,
    )


class RagPipeline:
    """Question in, grounded answer with page citations out."""

    def __init__(self, retriever: HybridRetriever, llm: LLM) -> None:
        self._retriever = retriever
        self._llm = llm

    def ask(self, question: str, mode: Mode = "hybrid") -> Answer:
        chunks = self._retriever.retrieve(question, mode)
        return answer_question(question, chunks, self._llm)
