"""One tool invocation made by a graph node."""

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank


class ToolCall(MeridianModel):
    """Accounting for a single read-only tool call."""

    node: NonBlank = Field(description="Graph node that made the call.")
    tool: NonBlank = Field(description="Tool name.")
    args_hash: NonBlank = Field(description="SHA-256 of the canonical JSON arguments.")
    ref_count: int = Field(ge=0, description="How many stable refs the response carried.")
    latency_ms: int = Field(ge=0, description="Wall time of the call, in milliseconds.")
    ok: bool = Field(description="True when the tool returned a typed payload.")
