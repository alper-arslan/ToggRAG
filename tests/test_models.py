import pytest
from pydantic import ValidationError

from toggrag.models import Chunk, Heading, Page


@pytest.fixture
def chunk():
    chunk = Chunk(
        id="c1",
        text="Far ayarı için ...",
        pages=[12, 13],
        section_path=["Aydınlatma", "Far Ayarı"],
        content_type="procedure",
    )
    return chunk


class TestChunk:
    def test_chunk_holds_text_and_source_data(self, chunk):
        assert chunk.pages == [12, 13]

    def test_text_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            Chunk(
                id="c1",
                text="",
                pages=[12, 13],
                section_path=["Aydınlatma", "Far Ayarı"],
                content_type="procedure",
            )

    @pytest.mark.parametrize("pages", [[], [0], [-1]])
    def test_pages_must_be_a_non_empty_list_of_positive_numbers(self, pages):
        valid_fields = {
            "id": "c1",
            "text": "Far ayarı için ...",
            "pages": pages,
            "section_path": ["Aydınlatma", "Far Ayarı"],
            "content_type": "procedure",
        }
        with pytest.raises(ValidationError):
            Chunk(**valid_fields)

    def test_happy_path(self, chunk):
        assert chunk.content_type == "procedure"
        assert chunk.section_path == ["Aydınlatma", "Far Ayarı"]

    def test_unknown_content_type_is_rejected(self):
        with pytest.raises(ValidationError):
            Chunk(
                id="c1",
                text="Far ayarı için ...",
                pages=[12, 13],
                section_path=["Aydınlatma", "Far Ayarı"],
                content_type="bad content_type",
            )

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValidationError):
            Chunk(
                id="c1",
                text="Far ayarı için ...",
                pages=[12, 13],
                section_path=["Aydınlatma", "Far Ayarı"],
                content_type="table",
                extra_field="This should raise ValidationError",
            )


def test_pages():
    # pages should be positive integer
    with pytest.raises(ValidationError):
        Page(text="Some text", page=0)
    with pytest.raises(ValidationError):
        Page(text="Some text", page=-1)
    # should accept empty text
    page = Page(text="", page=13)
    assert page.text == ""
    # happy path
    page = Page(text="Some text", page=13)
    assert page.text == "Some text"
    assert page.page == 13


def test_page_headers():
    # level should be positive integer
    with pytest.raises(ValidationError):
        Heading(level=0, title="Some Title", page=13)
    with pytest.raises(ValidationError):
        Heading(level=-1, title="Some Title", page=13)
    # Empty title rejected
    with pytest.raises(ValidationError):
        Heading(level=1, title="", page=13)

    # Happy path
    heading = Heading(level=3, title="Some Title", page=13)
    assert heading.level == 3
    assert heading.title == "Some Title"
    assert heading.page == 13
