"""Hypothesis property tests for Incident and MeridianState."""

import math
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from meridian.contracts import (
    Evidence,
    FaultType,
    Hypothesis,
    Incident,
    MeridianState,
    Patch,
    Postmortem,
    Severity,
    Verdict,
    VerificationResult,
)
from meridian.contracts.approval import ApprovalDecision, ApprovalOutcome

_WORDS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12)
_UTC = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 1, 1),
    timezones=st.just(UTC),
)
_NAIVE = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 1, 1),
)
_UNIT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_MONEY = st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
_DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"


def _nudge_off_integer(value: float) -> float:
    if math.isfinite(value) and value.is_integer():
        return value + 0.5
    return value


_CELL = st.one_of(
    _WORDS,
    st.integers(min_value=-1_000, max_value=1_000),
    st.booleans(),
    st.none(),
    st.floats(allow_nan=False, allow_infinity=False, width=32).map(_nudge_off_integer),
)
_ROW = st.dictionaries(_WORDS, _CELL, max_size=4)


@st.composite
def incidents(draw: st.DrawFn) -> Incident:
    return Incident(
        id=draw(st.uuids()),
        detected_at=draw(_UTC),
        dataset=draw(_WORDS),
        partition=draw(_WORDS),
        contract_name=draw(_WORDS),
        fault_type=draw(st.sampled_from(list(FaultType))),
        failing_checks=draw(st.lists(_WORDS, min_size=1, max_size=3)),
        sample_failure_rows=draw(st.lists(_ROW, max_size=50)),
        severity=draw(st.one_of(st.none(), st.sampled_from(list(Severity)))),
        blast_radius=draw(st.lists(_WORDS, max_size=3)),
    )


@st.composite
def consistent_states(draw: st.DrawFn) -> MeridianState:
    """Build a partial state whose parent ids agree whenever both sides exist."""
    incident = draw(st.one_of(st.none(), incidents()))
    hypothesis: Hypothesis | None = None
    if draw(st.booleans()):
        hypothesis = Hypothesis(
            id=draw(st.uuids()),
            incident_id=incident.id if incident is not None else draw(st.uuids()),
            root_cause_summary=draw(_WORDS),
            evidence=[Evidence(source=draw(_WORDS), reference=draw(_WORDS), excerpt=draw(_WORDS))]
            if draw(st.booleans())
            else [],
            confidence=draw(_UNIT),
        )
    patch: Patch | None = None
    if draw(st.booleans()):
        patch = Patch(
            id=draw(st.uuids()),
            incident_id=incident.id if incident is not None else draw(st.uuids()),
            hypothesis_id=hypothesis.id if hypothesis is not None else draw(st.uuids()),
            diff=_DIFF,
            regression_test=draw(_WORDS),
            files_touched=[f"{draw(_WORDS)}.py"],
            author_model=draw(_WORDS),
        )
    verdict: Verdict | None = None
    if draw(st.booleans()):
        verdict = Verdict(
            patch_id=patch.id if patch is not None else draw(st.uuids()),
            reviewer_model=draw(_WORDS),
            approved=draw(st.booleans()),
            objections=draw(st.lists(_WORDS, max_size=2)),
            round=draw(st.integers(min_value=1, max_value=3)),
        )
    verification: VerificationResult | None = None
    if draw(st.booleans()):
        passed = draw(st.booleans())
        failures = 0 if passed else draw(st.integers(min_value=1, max_value=5))
        verification = VerificationResult(
            patch_id=patch.id if patch is not None else draw(st.uuids()),
            passed=passed,
            rows_checked=draw(st.integers(min_value=0, max_value=100)),
            failures_remaining=failures,
            duration_ms=draw(st.integers(min_value=0, max_value=10_000)),
        )
    approval: ApprovalDecision | None = None
    if draw(st.booleans()):
        approval = ApprovalDecision(
            patch_id=patch.id if patch is not None else draw(st.uuids()),
            decision=draw(st.sampled_from(list(ApprovalOutcome))),
            approver=draw(_WORDS),
            decided_at=draw(_UTC),
            comment=draw(st.text(max_size=20)),
        )
    postmortem: Postmortem | None = None
    if draw(st.booleans()):
        postmortem = Postmortem(
            incident_id=incident.id if incident is not None else draw(st.uuids()),
            timeline=[],
            root_cause=draw(_WORDS),
            fix_summary=draw(_WORDS),
            time_to_resolution_s=draw(st.integers(min_value=0, max_value=10_000)),
            cost_usd=draw(_MONEY),
        )
    return MeridianState(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        verdict=verdict,
        verification=verification,
        approval=approval,
        postmortem=postmortem,
        fix_attempts=draw(st.integers(min_value=0, max_value=3)),
        critic_rounds=draw(st.integers(min_value=0, max_value=3)),
    )


@settings(max_examples=40, deadline=None, derandomize=True)
@given(incident=incidents())
def test_incident_properties(incident: Incident) -> None:
    assert incident.detected_at.tzinfo is UTC
    assert incident.detected_at.utcoffset() == timedelta(0)
    assert len(incident.failing_checks) >= 1
    assert len(incident.sample_failure_rows) <= 50
    restored = Incident.model_validate_json(incident.model_dump_json())
    assert restored == incident


@settings(max_examples=25, deadline=None, derandomize=True)
@given(stamp=_NAIVE, incident=incidents())
def test_incident_rejects_naive_timestamps(stamp: datetime, incident: Incident) -> None:
    payload = incident.model_dump()
    payload["detected_at"] = stamp
    with pytest.raises(ValidationError):
        Incident.model_validate(payload)


@settings(max_examples=20, deadline=None, derandomize=True)
@given(
    incident=incidents(),
    hours=st.integers(min_value=-12, max_value=14).filter(lambda value: value != 0),
)
def test_incident_rejects_non_utc_timestamps(incident: Incident, hours: int) -> None:
    shifted = incident.detected_at.replace(tzinfo=timezone(timedelta(hours=hours)))
    payload = incident.model_dump()
    payload["detected_at"] = shifted
    with pytest.raises(ValidationError):
        Incident.model_validate(payload)


@settings(max_examples=15, deadline=None, derandomize=True)
@given(rows=st.lists(_ROW, min_size=51, max_size=51), incident=incidents())
def test_incident_rejects_oversized_samples(
    rows: list[dict[str, object]],
    incident: Incident,
) -> None:
    payload = incident.model_dump()
    payload["sample_failure_rows"] = rows
    with pytest.raises(ValidationError):
        Incident.model_validate(payload)


@settings(max_examples=30, deadline=None, derandomize=True)
@given(state=consistent_states())
def test_meridian_state_properties(state: MeridianState) -> None:
    assert 0 <= state.fix_attempts <= 3
    assert 0 <= state.critic_rounds <= 3
    if state.incident is not None and state.hypothesis is not None:
        assert state.hypothesis.incident_id == state.incident.id
    if state.incident is not None and state.patch is not None:
        assert state.patch.incident_id == state.incident.id
    if state.hypothesis is not None and state.patch is not None:
        assert state.patch.hypothesis_id == state.hypothesis.id
    if state.patch is not None and state.verdict is not None:
        assert state.verdict.patch_id == state.patch.id
    if state.patch is not None and state.verification is not None:
        assert state.verification.patch_id == state.patch.id
    if state.patch is not None and state.approval is not None:
        assert state.approval.patch_id == state.patch.id
    if state.incident is not None and state.postmortem is not None:
        assert state.postmortem.incident_id == state.incident.id
    restored = MeridianState.model_validate_json(state.model_dump_json())
    assert restored == state


@settings(max_examples=20, deadline=None, derandomize=True)
@given(incident=incidents(), other=st.uuids())
def test_state_rejects_mismatched_hypothesis_parent(incident: Incident, other: UUID) -> None:
    assume(other != incident.id)
    hypothesis = Hypothesis(
        id=other,
        incident_id=other,
        root_cause_summary="drift",
        confidence=0.4,
    )
    with pytest.raises(ValidationError, match=r"hypothesis.incident_id"):
        MeridianState(incident=incident, hypothesis=hypothesis)


@settings(max_examples=10, deadline=None, derandomize=True)
@given(extra=st.integers(min_value=4, max_value=12))
def test_state_rejects_counters_over_the_loop_limits(extra: int) -> None:
    with pytest.raises(ValidationError):
        MeridianState(fix_attempts=extra)
    with pytest.raises(ValidationError):
        MeridianState(critic_rounds=extra)
