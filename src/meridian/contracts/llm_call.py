"""One model call made by a graph node. Phase 7 prices these rows."""

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank


class LLMCall(MeridianModel):
    """Token and latency accounting for a single model attempt."""

    node: NonBlank = Field(description="Graph node that made the call.")
    model: NonBlank = Field(description="Model name from configuration, or the fake model name.")
    prompt_version: NonBlank = Field(description="Prompt file version sent with the call.")
    input_tokens: int = Field(ge=0, description="Input tokens reported by the response.")
    output_tokens: int = Field(ge=0, description="Output tokens reported by the response.")
    latency_ms: int = Field(ge=0, description="Wall time of the attempt, in milliseconds.")
    attempt: int = Field(ge=1, description="1-based attempt number for this incident.")
    succeeded: bool = Field(description="True when structured output validated.")
