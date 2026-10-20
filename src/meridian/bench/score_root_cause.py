"""Score a root-cause hypothesis against the injected fault. Phase 6 will aggregate this."""

from pydantic import BaseModel, ConfigDict, Field

from meridian.contracts.hypothesis import Hypothesis
from meridian.contracts.injected_fault import InjectedFault


class RootCauseScore(BaseModel):
    """Two checks a later benchmark can average. This module does not claim an accuracy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    causal_event_correct: bool = Field(description="True when the causal event ids match.")
    columns_jaccard: float = Field(
        description="Jaccard overlap of suspect columns and columns affected.",
    )


def score(hypothesis: Hypothesis, injected_fault: InjectedFault) -> RootCauseScore:
    """Compare one hypothesis to the fault that was injected."""
    left = set(hypothesis.suspect_columns)
    right = set(injected_fault.columns_affected)
    union = left | right
    overlap = 1.0 if not union else len(left & right) / len(union)
    return RootCauseScore(
        causal_event_correct=hypothesis.causal_event_id == injected_fault.causal_event_id,
        columns_jaccard=overlap,
    )
