"""One model call made by a graph node. Phase 7 prices these rows."""

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank


class LLMCall(MeridianModel):
    """Token and latency accounting for a single model attempt."""

    node: NonBlank = Field(description="Graph node that made the call.")
    model: NonBlank = Field(description="Model id that answered, or the fake model name.")
    provider: NonBlank = Field(
        default="fake",
        description="Provider that answered: ollama, omniroute, anthropic, xai, or fake.",
    )
    prompt_version: NonBlank = Field(description="Prompt file version sent with the call.")
    input_tokens: int = Field(ge=0, description="Input tokens reported by the response.")
    output_tokens: int = Field(ge=0, description="Output tokens reported by the response.")
    prompt_tokens_before: int = Field(
        default=0,
        ge=0,
        description="Estimated input tokens before the sample cap and tool-summary shrink.",
    )
    prompt_tokens_after: int = Field(
        default=0,
        ge=0,
        description="Estimated input tokens of the prompt actually sent.",
    )
    latency_ms: int = Field(ge=0, description="Wall time of the attempt, in milliseconds.")
    attempt: int = Field(ge=1, description="1-based attempt number for this incident.")
    succeeded: bool = Field(description="True when structured output validated.")
    fallback_used: bool = Field(
        default=False,
        description="True when an earlier hop in the node's chain failed over.",
    )
    structured: bool = Field(
        default=False,
        description="True when this attempt asked the model to submit the node contract.",
    )
