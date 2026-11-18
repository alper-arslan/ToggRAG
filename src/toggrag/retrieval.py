from typing import Literal

from toggrag.bm25 import BM25Index
from toggrag.config import RerankerConfig, RetrievalConfig
from toggrag.embeddings import Embedder
from toggrag.fusion import rrf
from toggrag.models import Chunk
from toggrag.rerank import Reranker
from toggrag.store import VectorStore
from toggrag.text import analyze

Mode = Literal["bm25", "dense", "hybrid"]


class HybridRetriever:
    """BM25 + dense search fused with RRF, optionally reranked by a cross-encoder.

    `mode` lets the eval harness compare the parts on their own.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Index,
        store: VectorStore,
        embedder: Embedder,
        config: RetrievalConfig,
        reranker: Reranker | None = None,
        reranker_config: RerankerConfig | None = None,
    ) -> None:
        self._chunks = chunks
        self._bm25 = bm25
        self._store = store
        self._embedder = embedder
        self._config = config
        self._reranker = reranker
        self._reranker_config = reranker_config or RerankerConfig()

    def retrieve(self, query: str, mode: Mode = "hybrid") -> list[Chunk]:
        """Return the best chunks for the query, best first."""
        rankings: list[list[str]] = []
        if mode in ("bm25", "hybrid"):
            hits = self._bm25.search(analyze(query), self._config.bm25_top_k)
            rankings.append([str(i) for i, _ in hits])
        if mode in ("dense", "hybrid"):
            vector = self._embedder.embed([query])[0]
            hits = self._store.search(vector, self._config.dense_top_k)
            rankings.append([str(i) for i, _ in hits])

        fused = [self._chunks[int(i)] for i, _ in rrf(rankings, self._config.rrf_k)]
        if self._reranker and fused:
            candidates = fused[: self._reranker_config.candidates]
            return self._reranker.rerank(query, candidates, self._reranker_config.top_n)
        return fused[: self._config.final_top_k]
