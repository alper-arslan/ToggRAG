from toggrag.evaluation import (
    Question,
    evaluate,
    format_table,
    hit_at_k,
    judge_faithfulness,
    load_questions,
    recall_at_k,
    reciprocal_rank,
)
from toggrag.models import Chunk


def _chunk(*pages):
    return Chunk(
        id="c", text="t", pages=list(pages), section_path=["1"], content_type="text"
    )


def test_hit_at_k_only_looks_at_the_top_k():
    retrieved = [_chunk(1), _chunk(9)]
    assert hit_at_k(retrieved, [9], 1) == 0.0
    assert hit_at_k(retrieved, [9], 2) == 1.0


def test_recall_at_k_counts_covered_pages():
    retrieved = [_chunk(4, 5), _chunk(9)]
    assert recall_at_k(retrieved, [5, 6], 2) == 0.5


def test_reciprocal_rank_uses_first_relevant_chunk():
    retrieved = [_chunk(1), _chunk(2), _chunk(9)]
    assert reciprocal_rank(retrieved, [9]) == 1 / 3
    assert reciprocal_rank(retrieved, [50]) == 0.0


def test_evaluate_averages_over_questions():
    questions = [Question(question="a", pages=[1]), Question(question="b", pages=[2])]
    retrieve = {"a": [_chunk(1)], "b": [_chunk(7)]}.__getitem__
    metrics = evaluate(retrieve, questions)
    assert metrics["Hit@1"] == 0.5
    assert metrics["MRR"] == 0.5


def test_load_questions_reads_yaml(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text("- question: Nasıl?\n  pages: [3, 4]\n", encoding="utf-8")
    assert load_questions(path) == [Question(question="Nasıl?", pages=[3, 4])]


def test_gold_set_in_repo_is_valid():
    from pathlib import Path

    assert load_questions(Path("eval/questions.yaml"))


class _Judge:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user):
        return self.reply


def test_faithfulness_judge_reads_verdict():
    chunks = [_chunk(1)]
    assert judge_faithfulness("q", "a", chunks, _Judge("SUPPORTED"))
    assert not judge_faithfulness("q", "a", chunks, _Judge("UNSUPPORTED"))


def test_format_table_has_one_row_per_config():
    table = format_table({"bm25": {"MRR": 0.5}, "hybrid": {"MRR": 0.75}})
    assert table.splitlines()[0].split() == ["config", "MRR"]
    assert table.splitlines()[2].split() == ["hybrid", "0.75"]
