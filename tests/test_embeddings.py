import json

import httpx

from toggrag.embeddings import OllamaEmbedder


def _embedder(handler):
    client = httpx.Client(
        base_url="http://ollama", transport=httpx.MockTransport(handler)
    )
    return OllamaEmbedder("bge-m3", "http://ollama", client=client)


def test_embed_posts_texts_and_returns_vectors():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/embed"
        assert body["model"] == "bge-m3"
        return httpx.Response(
            200, json={"embeddings": [[1.0, 0.0]] * len(body["input"])}
        )

    assert _embedder(handler).embed(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]


def test_embed_sends_large_inputs_in_batches():
    sizes = []

    def handler(request):
        batch = json.loads(request.content)["input"]
        sizes.append(len(batch))
        return httpx.Response(200, json={"embeddings": [[0.0]] * len(batch)})

    vectors = _embedder(handler).embed(["x"] * 70)
    assert len(vectors) == 70
    assert sizes == [32, 32, 6]


def test_embed_raises_on_server_error():
    import pytest

    def handler(request):
        return httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        _embedder(handler).embed(["a"])
