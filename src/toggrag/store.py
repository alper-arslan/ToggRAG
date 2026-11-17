from qdrant_client import QdrantClient, models

from toggrag.config import VectorStoreConfig
from toggrag.models import Chunk

_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


class VectorStore:
    """Dense search over chunk embeddings, stored in a local Qdrant collection.

    A point's id is the chunk's position in the chunk list, so a search result
    maps straight back to the chunk without a second lookup.
    """

    def __init__(
        self, config: VectorStoreConfig, client: QdrantClient | None = None
    ) -> None:
        self._config = config
        self._client = client or QdrantClient(path=config.path)

    def rebuild(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Drop the collection and store the chunks with their vectors."""
        name = self._config.collection_name
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
        self._client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=len(vectors[0]), distance=_DISTANCES[self._config.distance]
            ),
        )
        self._client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=i,
                    vector=vector,
                    payload={"chunk_id": chunk.id, "pages": chunk.pages},
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ],
        )

    def search(self, vector: list[float], top_k: int) -> list[tuple[int, float]]:
        """Chunk positions and similarity scores, best first."""
        response = self._client.query_points(
            self._config.collection_name, query=vector, limit=top_k
        )
        return [(int(point.id), point.score) for point in response.points]
