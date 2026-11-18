from typing import Protocol

from toggrag.config import RerankerConfig
from toggrag.models import Chunk


class Reranker(Protocol):
    """Reorders candidate chunks by how well each answers the query."""

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]: ...


class CrossEncoderReranker:
    """Cross-encoder reranker (optional dependency: sentence-transformers)."""

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is not installed; run `uv sync --extra rerank`"
            ) from error
        self._model = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        scores = self._model.predict([(query, chunk.text) for chunk in chunks])
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_n]]


def make_reranker(config: RerankerConfig) -> Reranker | None:
    """Build the reranker selected in config, or None when it is turned off."""
    if not config.enabled:
        return None
    return CrossEncoderReranker(config.model)
