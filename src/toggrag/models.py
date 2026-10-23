from typing import Annotated, Literal

from pydantic import Field, PositiveInt, StringConstraints

from toggrag.shared import StrictModel


class Chunk(StrictModel):
    id: str = Field(description="The ID of the chunk")
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = (
        Field(description="The text of the chunk")
    )
    pages: list[PositiveInt] = Field(min_length=1, description="The pages of the chunk")
    section_path: list[str] = Field(description="The sections of the chunk")
    content_type: Literal["procedure", "warning", "specification", "table"] = Field(
        description="The content type of the chunk"
    )
