"""Score a root-cause hypothesis against the injected fault. Phase 6 will aggregate this."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from meridian.contracts.grounding import event_uuid
from meridian.contracts.hypothesis import Hypothesis
from meridian.contracts.injected_fault import InjectedFault


class RootCauseScore(BaseModel):
    """Checks a later benchmark can average. This module does not claim an accuracy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_causal_correct: bool = Field(
        description="True when the model's causal id matched before the guardrail.",
    )
    causal_event_correct: bool = Field(
        description="True when the causal id that survived the guardrail matches.",
    )
    columns_jaccard: float = Field(
        description="Jaccard overlap of suspect columns and columns affected.",
    )


def score(hypothesis: Hypothesis, injected_fault: InjectedFault) -> RootCauseScore:
    """Compare one hypothesis to the fault that was injected."""
    left = set(hypothesis.suspect_columns)
    right = set(injected_fault.columns_affected)
    union = left | right
    overlap = 1.0 if not union else len(left & right) / len(union)
    expected = injected_fault.causal_event_id
    return RootCauseScore(
        raw_causal_correct=_matches(_raw_causal(hypothesis), expected),
        causal_event_correct=_matches(hypothesis.causal_event_id, expected),
        columns_jaccard=overlap,
    )


def _raw_causal(hypothesis: Hypothesis) -> UUID | None:
    """The id the model named. A cleared id is recovered from its violation."""
    for finding in hypothesis.grounding_findings:
        if finding.field != "causal_event_id":
            continue
        parsed = event_uuid(finding.raw)
        if parsed is not None:
            return parsed
        return event_uuid(finding.normalized)
    return hypothesis.causal_event_id


def _matches(found: UUID | None, expected: UUID) -> bool:
    return found is not None and found == expected
