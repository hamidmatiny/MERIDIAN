"""The UNIT_MISMATCH live trace: event:<id> and a bare id are the same event."""

import json
from uuid import UUID, uuid4

from meridian.agents.root_cause import collect_refs, ground
from meridian.bench.score_root_cause import score
from meridian.contracts.fault_type import FaultType
from meridian.contracts.grounding import GroundingViolation
from meridian.contracts.hypothesis import Evidence, Hypothesis
from meridian.contracts.injected_fault import InjectedFault

_EVENT = "7288021c-eb08-59de-881b-d91cf150f43f"
_EVENT_ID = UUID(_EVENT)


def _hypothesis(**updates: object) -> Hypothesis:
    base = Hypothesis(
        id=uuid4(),
        incident_id=uuid4(),
        root_cause_summary="A CONFIG_CHANGE changed avg_speed_mps source_unit.",
        evidence=[
            Evidence(
                source="events",
                reference=f"event:{_EVENT}",
                excerpt="avg_speed_mps: source_unit changed",
            )
        ],
        confidence=0.85,
        suspect_columns=["avg_speed_mps"],
        mechanism="Speeds are no longer in metres per second.",
    )
    return base.model_copy(update=updates)


def _label() -> InjectedFault:
    return InjectedFault(
        id=uuid4(),
        fault_type=FaultType.UNIT_MISMATCH,
        dataset="trip_summary",
        columns_affected=["avg_speed_mps"],
        rows_affected=10,
        intensity=0.1,
        seed=7,
        causal_event_id=_EVENT_ID,
    )


def _list_events_payload(*, include_ref: bool) -> str:
    event: dict[str, object] = {
        "id": _EVENT,
        "ts": "2026-10-19T15:12:00Z",
        "kind": "CONFIG_CHANGE",
        "dataset": "trip_summary",
        "author": "pipeline-bot",
        "summary": "avg_speed_mps: source_unit changed",
        "diff_excerpt": "avg_speed_mps: source_unit changed",
    }
    if include_ref:
        event["ref"] = f"event:{_EVENT}"
    return json.dumps({"ref": "events:trip_summary", "events": [event]})


def test_unit_mismatch_trace_grounds_an_event_prefix() -> None:
    refs = set(collect_refs(_list_events_payload(include_ref=True)))
    grounded = ground(_hypothesis(), refs, causal_raw=f"event:{_EVENT}")
    assert grounded.causal_event_id == _EVENT_ID
    assert grounded.grounding_violations == 0
    assert grounded.evidence[0].reference == f"event:{_EVENT}"
    scored = score(grounded, _label())
    assert scored.raw_causal_correct is True
    assert scored.causal_event_correct is True


def test_list_events_id_grounds_without_get_event() -> None:
    payload = json.dumps({"events": [{"id": _EVENT, "dataset": "trip_summary"}]})
    refs = set(collect_refs(payload))
    grounded = ground(_hypothesis(), refs, causal_raw=_EVENT)
    assert f"event:{_EVENT}" in refs
    assert grounded.causal_event_id == _EVENT_ID
    assert grounded.grounding_violations == 0


def test_a_miss_records_the_raw_value_and_the_ref_sample() -> None:
    refs = set(collect_refs(_list_events_payload(include_ref=True)))
    missed = ground(
        _hypothesis(
            evidence=[
                Evidence(
                    source="events",
                    reference="event:00000000-0000-4000-8000-000000000099",
                    excerpt="invented",
                )
            ]
        ),
        refs,
        causal_raw="event:00000000-0000-4000-8000-000000000099",
    )
    assert missed.causal_event_id is None
    assert missed.evidence == []
    assert missed.grounding_violations == 2
    causal = next(item for item in missed.grounding_findings if item.field == "causal_event_id")
    assert causal.raw == "event:00000000-0000-4000-8000-000000000099"
    assert causal.normalized == "event:00000000-0000-4000-8000-000000000099"
    assert f"event:{_EVENT}" in causal.allowed_refs_sample
    scored = score(missed, _label())
    assert scored.raw_causal_correct is False
    assert scored.causal_event_correct is False


def test_score_keeps_a_correct_id_the_guardrail_cleared() -> None:
    cleared = _hypothesis(causal_event_id=None, evidence=[], confidence=0.6).model_copy(
        update={
            "grounding_violations": 1,
            "grounding_findings": [
                GroundingViolation(
                    field="causal_event_id",
                    raw=f"event:{_EVENT}",
                    normalized=f"event:{_EVENT}",
                    allowed_refs_sample=["events:trip_summary"],
                )
            ],
        }
    )
    scored = score(cleared, _label())
    assert scored.raw_causal_correct is True
    assert scored.causal_event_correct is False
