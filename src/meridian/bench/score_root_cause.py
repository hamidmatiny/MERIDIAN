"""Score a root-cause hypothesis against the injected fault. Phase 6 will aggregate this."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from meridian.contracts.fault_type import FaultType
from meridian.contracts.grounding import event_uuid
from meridian.contracts.hypothesis import Hypothesis
from meridian.contracts.injected_fault import InjectedFault

_RENAMED = "_renamed"


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
    left = _named(hypothesis.suspect_columns, injected_fault)
    right = _expected(injected_fault)
    union = left | right
    overlap = 1.0 if not union else len(left & right) / len(union)
    expected = injected_fault.causal_event_id
    return RootCauseScore(
        raw_causal_correct=_matches(_raw_causal(hypothesis), expected),
        causal_event_correct=_matches(hypothesis.causal_event_id, expected),
        columns_jaccard=overlap,
    )


def _expected(injected_fault: InjectedFault) -> set[str]:
    """Columns the hypothesis should name. A rename pair counts as one column."""
    names = list(injected_fault.columns_affected)
    if injected_fault.fault_type is not FaultType.SCHEMA_DRIFT:
        return set(names)
    present = set(names)
    paired: set[str] = set()
    expected: set[str] = set()
    for name in names:
        renamed = name + _RENAMED
        if renamed in present:
            expected.add(name)
            paired.add(name)
            paired.add(renamed)
    for name in names:
        if name not in paired:
            expected.add(name)
    return expected


def _named(suspects: list[str], injected_fault: InjectedFault) -> set[str]:
    """Map a renamed column back to the original when the label records the pair."""
    if injected_fault.fault_type is not FaultType.SCHEMA_DRIFT:
        return set(suspects)
    expected = _expected(injected_fault)
    found: set[str] = set()
    for name in suspects:
        if name.endswith(_RENAMED):
            original = name[: -len(_RENAMED)]
            if original in expected:
                found.add(original)
                continue
        found.add(name)
    return found


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
