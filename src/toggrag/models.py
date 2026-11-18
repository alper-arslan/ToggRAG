from typing import Annotated, Literal

from pydantic import Field, PositiveInt, StringConstraints

from toggrag.shared import StrictModel


class Chunk(StrictModel):
    """A retrievable piece of the manual, with the metadata needed to cite it."""

    id: str = Field(description="Unique identifier of the chunk")
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = (
        Field(description="Chunk text, stripped of surrounding whitespace; never empty")
    )
    pages: list[PositiveInt] = Field(
        min_length=1,
        description="Page numbers (1-based, as shown by a PDF viewer) the chunk spans",
    )
    section_path: list[str] = Field(
        description="Heading hierarchy from chapter down to subsection"
    )
    content_type: Literal["procedure", "warning", "specification", "table", "text"] = (
        Field(
            description="Kind of content, used to boost procedures for how-to questions"
        )
    )


class Page(StrictModel):
    """Raw text of one PDF page."""

    page: PositiveInt = Field(
        description="Page number (1-based, as shown by a PDF viewer)"
    )
    text: str = Field(
        description="Extracted text; may be empty for image-only pages such as covers"
    )


class Heading(StrictModel):
    """One entry of the PDF's bookmark outline."""

    level: PositiveInt = Field(
        description="Depth in the outline: 1 is a chapter, 2 a section, 3 a subsection"
    )
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = (
        Field(description="Heading text, e.g. '2.1.1 Güvenlik Talimatları'")
    )
    page: PositiveInt = Field(
        description="Page number (1-based) on which the heading appears"
    )


class Citation(StrictModel):
    """A source the answer relies on."""

    number: PositiveInt = Field(description="Number used in the answer text, e.g. [2]")
    pages: list[PositiveInt] = Field(
        min_length=1, description="Manual pages of the source"
    )
    section_path: list[str] = Field(description="Heading hierarchy of the source")


class Answer(StrictModel):
    """A grounded answer to a question, with the sources it cites."""

    text: str = Field(description="Answer text; cites sources as [1], [2], ...")
    citations: list[Citation] = Field(
        description="Sources actually cited in the text; empty when the manual has no answer"
    )
