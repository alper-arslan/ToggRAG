from toggrag.bm25 import BM25Index
from toggrag.text import analyze

DOCS = [
    "Lastik basıncı kontrol edilir",
    "Silecek suyu doldurma",
    "Lastikler ve jantlar değiştirilir",
]


def _index():
    return BM25Index([analyze(doc) for doc in DOCS])


def test_matching_document_ranks_first():
    hits = _index().search(analyze("silecek"), top_k=3)
    assert hits[0][0] == 1


def test_stemming_matches_suffix_variants():
    hits = _index().search(analyze("lastiklerin"), top_k=3)
    assert {i for i, _ in hits} == {0, 2}


def test_unknown_terms_return_nothing():
    assert _index().search(analyze("fren"), top_k=3) == []


def test_top_k_limits_results():
    assert len(_index().search(analyze("lastik"), top_k=1)) == 1


def test_rarer_term_scores_higher():
    index = BM25Index([["a", "b"], ["a", "c"], ["a", "d"]])
    assert index.scores(["b"])[0] > index.scores(["a"])[0]


def test_empty_index():
    assert BM25Index([]).search(["a"], top_k=3) == []
