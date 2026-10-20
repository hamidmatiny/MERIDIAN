"""Shared model configuration for every inter-agent contract."""

from pydantic import BaseModel, ConfigDict


class MeridianModel(BaseModel):
    """Frozen strict base. Unknown fields are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
