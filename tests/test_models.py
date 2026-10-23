import pytest
from pydantic import ValidationError

from toggrag.models import Chunk


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
