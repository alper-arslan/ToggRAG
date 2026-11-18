from toggrag.models import Answer, Citation
from toggrag.ui import format_answer


def test_format_answer_lists_cited_pages():
    answer = Answer(
        text="Kapağı açın [1].",
        citations=[
            Citation(number=1, pages=[4, 5], section_path=["1 Giriş", "1.1 Alt"])
        ],
    )
    assert format_answer(answer) == (
        "Kapağı açın [1].\n\n**Kaynaklar**\n- [1] sayfa 4, 5: 1 Giriş > 1.1 Alt"
    )


def test_format_answer_without_citations_is_just_the_text():
    assert format_answer(Answer(text="Bilgi yok.", citations=[])) == "Bilgi yok."
