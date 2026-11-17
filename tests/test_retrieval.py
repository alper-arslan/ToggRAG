import pytest
from qdrant_client import QdrantClient

from toggrag.bm25 import BM25Index
from toggrag.config import RerankerConfig, RetrievalConfig, VectorStoreConfig
from toggrag.models import Chunk
from toggrag.retrieval import HybridRetriever
from toggrag.store import VectorStore
from toggrag.text import analyze

TEXTS = [
    "Silecek suyu deposunu doldurun",
    "Lastik basıncını kontrol edin",
    "Park sensörlerini açma ve kapatma",
]
VECTORS = {
    "wiper": [1.0, 0.0, 0.0],
    "tyre": [0.0, 1.0, 0.0],
    "park": [0.0, 0.0, 1.0],
}


class FakeEmbedder:
    """Maps a query to a fixed vector by keyword, standing in for a real model."""

    def embed(self, texts):
        out = []
        for text in texts:
            key = "park" if "otopark" in text else "wiper"
            out.append(VECTORS[key])
        return out


class ReverseReranker:
    def rerank(self, query, chunks, top_n):
        return list(reversed(chunks))[:top_n]


def _chunks():
    return [
        Chunk(
            id=f"c{i}",
            text=text,
            pages=[i + 1],
            section_path=["1"],
            content_type="text",
        )
        for i, text in enumerate(TEXTS)
    ]


@pytest.fixture
def retriever():
    chunks = _chunks()
    store = VectorStore(VectorStoreConfig(), client=QdrantClient(":memory:"))
    store.rebuild(chunks, list(VECTORS.values()))
    bm25 = BM25Index([analyze(chunk.text) for chunk in chunks])
    return HybridRetriever(
        chunks, bm25, store, FakeEmbedder(), RetrievalConfig(final_top_k=2)
    )


def test_bm25_mode_uses_keywords_only(retriever):
    assert retriever.retrieve("lastik", mode="bm25")[0].id == "c1"


def test_dense_mode_uses_vectors_only(retriever):
    assert retriever.retrieve("otopark", mode="dense")[0].id == "c2"


def test_hybrid_finds_what_either_side_finds(retriever):
    ids = [c.id for c in retriever.retrieve("lastik otopark")]
    assert ids[0] in {"c1", "c2"} and len(ids) == 2


def test_final_top_k_limits_results(retriever):
    assert len(retriever.retrieve("lastik")) <= 2


def test_reranker_reorders_candidates():
    chunks = _chunks()
    store = VectorStore(VectorStoreConfig(), client=QdrantClient(":memory:"))
    store.rebuild(chunks, list(VECTORS.values()))
    bm25 = BM25Index([analyze(chunk.text) for chunk in chunks])
    retriever = HybridRetriever(
        chunks,
        bm25,
        store,
        FakeEmbedder(),
        RetrievalConfig(),
        reranker=ReverseReranker(),
        reranker_config=RerankerConfig(candidates=3, top_n=1),
    )
    query = "silecek lastik"
    fused = retriever._bm25.search(analyze(query), 3)
    best_first = [chunks[i].id for i, _ in fused]
    reranked = retriever.retrieve(query, mode="bm25")
    assert [c.id for c in reranked] == [best_first[-1]]
