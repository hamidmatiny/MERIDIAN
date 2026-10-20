"""Valid construction and validator rejection for every contract."""

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError
from tests.contracts.builders import (
    DIFF,
    LATER,
    NOW,
    REGRESSION,
    make_approval,
    make_evidence,
    make_hypothesis,
    make_incident,
    make_patch,
    make_postmortem,
    make_verdict,
    make_verification,
)

from meridian.contracts import (
    MAX_CRITIC_ROUNDS,
    MAX_FIX_ATTEMPTS,
    ApprovalOutcome,
    FaultType,
    Hypothesis,
    Incident,
    MeridianState,
    Patch,
    Postmortem,
    Severity,
    TimelineEvent,
    Verdict,
    VerificationResult,
)
from meridian.schema_export import CONTRACT_SCHEMAS


def test_each_enum_value_round_trips_as_itself() -> None:
    assert list(Severity) == [Severity.SEV1, Severity.SEV2, Severity.SEV3, Severity.SEV4]
    assert {item.value for item in FaultType} == {
        "SCHEMA_DRIFT",
        "NULL_EXPLOSION",
        "UNIT_MISMATCH",
        "DUPLICATE_KEYS",
        "LATE_PARTITION",
        "DISTRIBUTION_SHIFT",
    }
    assert list(ApprovalOutcome) == [
        ApprovalOutcome.APPROVED,
        ApprovalOutcome.REJECTED,
        ApprovalOutcome.TIMEOUT,
    ]


def test_models_are_frozen_and_strict() -> None:
    for _stem, source in CONTRACT_SCHEMAS:
        if not isinstance(source, type) or not issubclass(source, BaseModel):
            continue
        assert source.model_config["frozen"] is True
        assert source.model_config["strict"] is True
        assert source.model_config["extra"] == "forbid"


def test_valid_incident_allows_unset_severity() -> None:
    incident = make_incident()
    assert incident.severity is None
    assert incident.detected_at.tzinfo is UTC
    assert incident.fault_type is FaultType.NULL_EXPLOSION


def test_valid_incident_with_triage_fields() -> None:
    incident = make_incident(severity=Severity.SEV2, blast_radius=["billing", "dashboards"])
    assert incident.severity is Severity.SEV2
    assert incident.blast_radius == ["billing", "dashboards"]


def test_incident_accepts_explicit_utc_offset_and_normalizes() -> None:
    stamp = datetime(2026, 10, 19, 12, 0, tzinfo=timezone(timedelta(0)))
    incident = make_incident(detected_at=stamp)
    assert incident.detected_at.tzinfo is UTC


def test_incident_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match=r"timezone-aware UTC"):
        make_incident(detected_at=datetime(2026, 10, 19, 12, 0))


def test_incident_rejects_non_utc_datetime() -> None:
    stamp = datetime(2026, 10, 19, 12, 0, tzinfo=timezone(timedelta(hours=-3)))
    with pytest.raises(ValidationError, match=r"timestamp must be UTC"):
        make_incident(detected_at=stamp)


def test_incident_rejects_blank_names_and_empty_checks() -> None:
    with pytest.raises(ValidationError):
        Incident(
            id=uuid4(),
            detected_at=NOW,
            dataset="  ",
            partition="dt",
            contract_name="orders",
            fault_type=FaultType.SCHEMA_DRIFT,
            failing_checks=["null_rate"],
        )
    with pytest.raises(ValidationError):
        make_incident(failing_checks=[])
    with pytest.raises(ValidationError):
        make_incident(failing_checks=[" "])


def test_incident_rejects_unknown_fields_and_mutation() -> None:
    with pytest.raises(ValidationError):
        Incident(
            id=uuid4(),
            detected_at=NOW,
            dataset="orders",
            partition="dt",
            contract_name="orders",
            fault_type=FaultType.LATE_PARTITION,
            failing_checks=["freshness"],
            unexpected=True,
        )
    incident = make_incident()
    with pytest.raises(ValidationError):
        incident.dataset = "other"


def test_sample_rows_accept_fifty_and_reject_fifty_one() -> None:
    rows = [{"id": index} for index in range(50)]
    incident = make_incident(sample_failure_rows=rows)
    assert len(incident.sample_failure_rows) == 50
    with pytest.raises(ValidationError, match=r"50"):
        make_incident(sample_failure_rows=[{"id": index} for index in range(51)])


def test_valid_hypothesis_patch_verdict_and_verification() -> None:
    incident = make_incident(severity=Severity.SEV1)
    hypothesis = make_hypothesis(incident_id=incident.id, confidence=0.0)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    verdict = make_verdict(patch_id=patch.id, round_number=3)
    verification = make_verification(patch_id=patch.id, passed=False, failures_remaining=2)
    approval = make_approval(patch_id=patch.id)
    postmortem = make_postmortem(incident_id=incident.id)
    state = MeridianState(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        verdict=verdict,
        verification=verification,
        approval=approval,
        postmortem=postmortem,
        fix_attempts=MAX_FIX_ATTEMPTS,
        critic_rounds=MAX_CRITIC_ROUNDS,
    )
    assert state.hypothesis is not None
    assert state.hypothesis.evidence[0] == make_evidence()
    assert state.verdict is not None
    assert state.verdict.round == 3
    assert approval.decision is ApprovalOutcome.APPROVED
    assert postmortem.time_to_resolution_s == 300


def test_hypothesis_confidence_bounds() -> None:
    incident_id = uuid4()
    assert make_hypothesis(incident_id=incident_id, confidence=1).confidence == 1.0
    assert make_hypothesis(incident_id=incident_id, confidence=1.0).confidence == 1.0
    with pytest.raises(ValidationError):
        make_hypothesis(confidence=1.01)
    with pytest.raises(ValidationError):
        make_hypothesis(confidence=-0.01)
    with pytest.raises(ValidationError):
        make_hypothesis(confidence=float("nan"))
    with pytest.raises(ValidationError):
        make_hypothesis(confidence=float("inf"))
    with pytest.raises(ValidationError):
        make_hypothesis(confidence=True)  # type: ignore[arg-type]


def test_hypothesis_rejects_blank_summary_and_evidence() -> None:
    with pytest.raises(ValidationError):
        Hypothesis(
            id=uuid4(),
            incident_id=uuid4(),
            root_cause_summary=" ",
            confidence=0.2,
        )
    with pytest.raises(ValidationError):
        make_evidence_blank()


def make_evidence_blank() -> None:
    from meridian.contracts import Evidence

    Evidence(source=" ", reference="ref", excerpt="")


def test_evidence_allows_empty_excerpt() -> None:
    evidence = make_evidence()
    assert evidence.excerpt == "null_rate = 0.5"
    empty = type(evidence)(source="logs", reference="run-1", excerpt="")
    assert empty.excerpt == ""


def test_verdict_round_limits() -> None:
    assert make_verdict(round_number=1).round == 1
    with pytest.raises(ValidationError):
        make_verdict(round_number=0)
    with pytest.raises(ValidationError):
        make_verdict(round_number=4)
    with pytest.raises(ValidationError):
        Verdict(
            patch_id=uuid4(),
            reviewer_model=" ",
            approved=True,
            round=1,
        )


def test_verification_rejects_pass_with_remaining_failures() -> None:
    make_verification(passed=True, failures_remaining=0)
    make_verification(passed=False, failures_remaining=0)
    with pytest.raises(ValidationError, match=r"failures_remaining"):
        make_verification(passed=True, failures_remaining=1)
    with pytest.raises(ValidationError):
        VerificationResult(
            patch_id=uuid4(),
            passed=False,
            rows_checked=-1,
            failures_remaining=1,
            duration_ms=0,
        )
    with pytest.raises(ValidationError):
        VerificationResult(
            patch_id=uuid4(),
            passed=False,
            rows_checked=1,
            failures_remaining=-1,
            duration_ms=0,
        )
    with pytest.raises(ValidationError):
        VerificationResult(
            patch_id=uuid4(),
            passed=False,
            rows_checked=1,
            failures_remaining=1,
            duration_ms=-1,
        )


def test_approval_outcomes_and_naive_timestamp() -> None:
    for outcome in ApprovalOutcome:
        decision = make_approval()
        replaced = decision.model_copy(update={"decision": outcome})
        assert replaced.decision is outcome
    with pytest.raises(ValidationError, match=r"timezone-aware UTC"):
        type(make_approval())(
            patch_id=uuid4(),
            decision=ApprovalOutcome.TIMEOUT,
            approver="U1",
            decided_at=datetime(2026, 10, 19, 12, 0),
            comment="",
        )
    with pytest.raises(ValidationError):
        type(make_approval())(
            patch_id=uuid4(),
            decision=ApprovalOutcome.REJECTED,
            approver=" ",
            decided_at=NOW,
            comment="",
        )


def test_postmortem_timeline_must_be_ordered() -> None:
    report = make_postmortem()
    assert [event.actor for event in report.timeline] == ["sentinel", "human"]
    same_time = Postmortem(
        incident_id=uuid4(),
        timeline=[
            TimelineEvent(occurred_at=NOW, actor="sentinel", summary="Detected."),
            TimelineEvent(occurred_at=NOW, actor="triage", summary="Ranked."),
        ],
        root_cause="cause",
        fix_summary="fix",
        time_to_resolution_s=0,
        cost_usd=0,
    )
    assert len(same_time.timeline) == 2
    empty = Postmortem(
        incident_id=uuid4(),
        timeline=[],
        root_cause="cause",
        fix_summary="fix",
        time_to_resolution_s=1,
        cost_usd=1,
    )
    assert empty.timeline == []
    with pytest.raises(ValidationError, match=r"ordered"):
        Postmortem(
            incident_id=uuid4(),
            timeline=[
                TimelineEvent(occurred_at=LATER, actor="human", summary="Approved."),
                TimelineEvent(occurred_at=NOW, actor="sentinel", summary="Detected."),
            ],
            root_cause="cause",
            fix_summary="fix",
            time_to_resolution_s=1,
            cost_usd=1.5,
        )
    with pytest.raises(ValidationError):
        Postmortem(
            incident_id=uuid4(),
            root_cause="cause",
            fix_summary="fix",
            time_to_resolution_s=-1,
            cost_usd=0.0,
        )
    with pytest.raises(ValidationError):
        Postmortem(
            incident_id=uuid4(),
            root_cause="cause",
            fix_summary="fix",
            time_to_resolution_s=1,
            cost_usd=-0.01,
        )
    with pytest.raises(ValidationError):
        Postmortem(
            incident_id=uuid4(),
            root_cause=" ",
            fix_summary="fix",
            time_to_resolution_s=1,
            cost_usd=0.0,
        )


def test_patch_requires_a_file_and_content() -> None:
    patch = make_patch()
    assert patch.author_model == "claude"
    with pytest.raises(ValidationError):
        Patch(
            id=uuid4(),
            incident_id=uuid4(),
            hypothesis_id=uuid4(),
            diff=" ",
            regression_test=REGRESSION,
            files_touched=["a.py"],
            author_model="claude",
        )
    with pytest.raises(ValidationError):
        Patch(
            id=uuid4(),
            incident_id=uuid4(),
            hypothesis_id=uuid4(),
            diff=DIFF,
            regression_test=" ",
            files_touched=["a.py"],
            author_model="claude",
        )
    with pytest.raises(ValidationError):
        Patch(
            id=uuid4(),
            incident_id=uuid4(),
            hypothesis_id=uuid4(),
            diff=DIFF,
            regression_test=REGRESSION,
            files_touched=[],
            author_model="claude",
        )


def test_empty_state_is_valid() -> None:
    state = MeridianState()
    assert state.incident is None
    assert state.fix_attempts == 0
    assert state.critic_rounds == 0


def test_loop_counters_accept_their_limits_and_reject_the_next_value() -> None:
    assert MAX_CRITIC_ROUNDS == 3
    assert MAX_FIX_ATTEMPTS == 3
    assert MeridianState(fix_attempts=3, critic_rounds=3).fix_attempts == 3
    with pytest.raises(ValidationError):
        MeridianState(critic_rounds=4)
    with pytest.raises(ValidationError):
        MeridianState(fix_attempts=4)
    with pytest.raises(ValidationError):
        MeridianState(critic_rounds=-1)
    with pytest.raises(ValidationError):
        MeridianState(fix_attempts=-1)


def test_parent_ids_must_match_when_both_sides_are_present() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    MeridianState(incident=incident, hypothesis=hypothesis, patch=patch)

    with pytest.raises(ValidationError, match=r"hypothesis.incident_id"):
        MeridianState(incident=incident, hypothesis=make_hypothesis())
    with pytest.raises(ValidationError, match=r"patch.incident_id"):
        MeridianState(incident=incident, patch=make_patch(hypothesis_id=hypothesis.id))
    with pytest.raises(ValidationError, match=r"patch.hypothesis_id"):
        MeridianState(hypothesis=hypothesis, patch=make_patch(incident_id=incident.id))
    with pytest.raises(ValidationError, match=r"verdict.patch_id"):
        MeridianState(patch=patch, verdict=make_verdict())
    with pytest.raises(ValidationError, match=r"verification.patch_id"):
        MeridianState(patch=patch, verification=make_verification())
    with pytest.raises(ValidationError, match=r"approval.patch_id"):
        MeridianState(patch=patch, approval=make_approval())
    with pytest.raises(ValidationError, match=r"postmortem.incident_id"):
        MeridianState(incident=incident, postmortem=make_postmortem())


def test_child_records_may_exist_before_their_parents() -> None:
    state = MeridianState(
        hypothesis=make_hypothesis(),
        verdict=make_verdict(),
        verification=make_verification(passed=False, failures_remaining=1),
        approval=make_approval(),
        postmortem=make_postmortem(),
    )
    assert state.incident is None
    assert state.patch is None
