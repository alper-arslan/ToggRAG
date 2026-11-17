from typing import Protocol

import httpx

from toggrag.config import EmbeddingConfig

_BATCH_SIZE = 32


class Embedder(Protocol):
    """Turns texts into dense vectors. Any provider that does this can be used."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Embeddings from an Ollama server (POST /api/embed)."""

    def __init__(
        self, model: str, base_url: str, client: httpx.Client | None = None
    ) -> None:
        self._model = model
        self._client = client or httpx.Client(base_url=base_url, timeout=120)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            response = self._client.post(
                "/api/embed",
                json={
                    "model": self._model,
                    "input": texts[start : start + _BATCH_SIZE],
                },
            )
            response.raise_for_status()
            vectors += response.json()["embeddings"]
        return vectors


class SentenceTransformerEmbedder:
    """Local embeddings with sentence-transformers (optional dependency)."""

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is not installed; run `uv sync --extra local`"
            ) from error
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def make_embedder(config: EmbeddingConfig) -> Embedder:
    """Build the embedder selected in config."""
    if config.provider == "ollama":
        return OllamaEmbedder(config.model, config.base_url)
    return SentenceTransformerEmbedder(config.model)
