from toggrag.fusion import rrf


def test_item_in_both_rankings_beats_item_in_one():
    fused = rrf([["a", "b"], ["c", "b"]], k=60)
    assert fused[0][0] == "b"


def test_scores_follow_the_formula():
    fused = dict(rrf([["a"], ["a"]], k=60))
    assert fused["a"] == 2 / 61


def test_lower_rank_scores_less():
    fused = dict(rrf([["a", "b"]], k=60))
    assert fused["a"] > fused["b"]


def test_empty_rankings():
    assert rrf([]) == []
    assert rrf([[], []]) == []
