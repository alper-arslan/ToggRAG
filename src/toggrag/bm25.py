import math
from collections import Counter


class BM25Index:
    """Okapi BM25 over pre-tokenized documents.

    Documents and queries must go through the same analyzer (text.analyze), so
    Turkish suffix variants of a word hit the same term.
    """

    def __init__(
        self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self._k1 = k1
        self._b = b
        self._frequencies = [Counter(document) for document in documents]
        self._lengths = [len(document) for document in documents]
        self._average_length = sum(self._lengths) / len(documents) if documents else 0
        containing = Counter(term for f in self._frequencies for term in f)
        count = len(documents)
        self._idf = {
            term: math.log(1 + (count - n + 0.5) / (n + 0.5))
            for term, n in containing.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        """BM25 score of every document for the query, in document order."""
        scores = []
        for frequencies, length in zip(self._frequencies, self._lengths):
            score = 0.0
            for term in set(query):
                tf = frequencies.get(term, 0)
                if tf:
                    norm = 1 - self._b + self._b * length / self._average_length
                    score += (
                        self._idf[term] * tf * (self._k1 + 1) / (tf + self._k1 * norm)
                    )
            scores.append(score)
        return scores

    def search(self, query: list[str], top_k: int) -> list[tuple[int, float]]:
        """Indexes and scores of the best documents; documents without a hit are skipped."""
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [(i, scores[i]) for i in ranked[:top_k] if scores[i] > 0]
