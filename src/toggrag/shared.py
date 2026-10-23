from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for our pydantic models: unknown fields raise instead of being ignored."""

    model_config = ConfigDict(extra="forbid")
