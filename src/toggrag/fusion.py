from collections import defaultdict


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    Each ranking is a list of ids, best first. An id scores 1 / (k + rank) in
    every ranking that contains it; the scores are summed. Only ranks are used,
    so BM25 and cosine scores need no normalisation. Ties keep first-seen order.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
