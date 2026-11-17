from toggrag.answer import (
    NOT_FOUND,
    answer_question,
    format_context,
    parse_citations,
)
from toggrag.models import Chunk


def _chunk(i, pages):
    return Chunk(
        id=f"c{i}",
        text=f"metin {i}",
        pages=pages,
        section_path=["1 Giriş", f"1.{i} Alt"],
        content_type="text",
    )


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


def test_format_context_numbers_chunks_with_pages_and_section():
    context = format_context([_chunk(1, [4, 5])])
    assert context == "[1] (Sayfa 4, 5; 1 Giriş > 1.1 Alt)\nmetin 1"


def test_parse_citations_keeps_order_and_drops_duplicates():
    chunks = [_chunk(1, [4]), _chunk(2, [9])]
    citations = parse_citations("Önce [2], sonra [1] ve yine [2].", chunks)
    assert [(c.number, c.pages) for c in citations] == [(2, [9]), (1, [4])]


def test_parse_citations_ignores_numbers_that_do_not_exist():
    assert parse_citations("Yanıt [7] ve [0].", [_chunk(1, [4])]) == []


def test_answer_cites_the_sources_it_used():
    llm = FakeLLM("Kapağı açın [2].")
    answer = answer_question("nasıl?", [_chunk(1, [4]), _chunk(2, [9])], llm)
    assert answer.text == "Kapağı açın [2]."
    assert [c.pages for c in answer.citations] == [[9]]


def test_answer_sends_question_and_context_to_the_llm():
    llm = FakeLLM("Tamam [1].")
    answer_question("Silecek nasıl?", [_chunk(1, [4])], llm)
    system, user = llm.calls[0]
    assert "YALNIZCA" in system
    assert "Soru: Silecek nasıl?" in user
    assert "[1] (Sayfa 4" in user


def test_refusal_has_no_citations():
    answer = answer_question("?", [_chunk(1, [4])], FakeLLM(NOT_FOUND))
    assert answer.text == NOT_FOUND
    assert answer.citations == []


def test_no_chunks_means_no_llm_call():
    llm = FakeLLM("bu çağrılmamalı")
    answer = answer_question("?", [], llm)
    assert answer.text == NOT_FOUND
    assert llm.calls == []
