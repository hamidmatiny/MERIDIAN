"""Factories for valid contract instances."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from meridian.contracts import (
    ApprovalDecision,
    ApprovalOutcome,
    Evidence,
    FaultType,
    Hypothesis,
    Incident,
    Patch,
    Postmortem,
    Severity,
    TimelineEvent,
    Verdict,
    VerificationResult,
)
from meridian.contracts.incident import RowValue

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 10, 19, 12, 5, tzinfo=UTC)

DIFF = "--- a/contracts/orders.py\n+++ b/contracts/orders.py\n@@ -1 +1 @@\n-old\n+new\n"
REGRESSION = "def test_orders_null_rate() -> None:\n    assert True\n"


def make_incident(
    *,
    incident_id: UUID | None = None,
    detected_at: datetime | None = None,
    sample_failure_rows: list[dict[str, RowValue]] | None = None,
    severity: Severity | None = None,
    failing_checks: list[str] | None = None,
    blast_radius: list[str] | None = None,
) -> Incident:
    return Incident(
        id=uuid4() if incident_id is None else incident_id,
        detected_at=NOW if detected_at is None else detected_at,
        dataset="orders",
        partition="dt=2026-10-19",
        contract_name="orders_daily",
        fault_type=FaultType.NULL_EXPLOSION,
        failing_checks=["null_rate"] if failing_checks is None else failing_checks,
        sample_failure_rows=(
            [{"order_id": "abc", "amount": 1, "ratio": 1.5, "flag": True, "note": None}]
            if sample_failure_rows is None
            else sample_failure_rows
        ),
        severity=severity,
        blast_radius=[] if blast_radius is None else blast_radius,
    )


def make_evidence() -> Evidence:
    return Evidence(source="git", reference="abc123", excerpt="null_rate = 0.5")


def make_hypothesis(
    *,
    hypothesis_id: UUID | None = None,
    incident_id: UUID | None = None,
    confidence: float = 0.75,
    evidence: list[Evidence] | None = None,
) -> Hypothesis:
    return Hypothesis(
        id=uuid4() if hypothesis_id is None else hypothesis_id,
        incident_id=uuid4() if incident_id is None else incident_id,
        root_cause_summary="An upstream cast widened the null rate.",
        evidence=[make_evidence()] if evidence is None else evidence,
        confidence=confidence,
    )


def make_patch(
    *,
    patch_id: UUID | None = None,
    incident_id: UUID | None = None,
    hypothesis_id: UUID | None = None,
) -> Patch:
    return Patch(
        id=uuid4() if patch_id is None else patch_id,
        incident_id=uuid4() if incident_id is None else incident_id,
        hypothesis_id=uuid4() if hypothesis_id is None else hypothesis_id,
        diff=DIFF,
        regression_test=REGRESSION,
        files_touched=["contracts/orders.py"],
        author_model="claude",
    )


def make_verdict(*, patch_id: UUID | None = None, round_number: int = 1) -> Verdict:
    return Verdict(
        patch_id=uuid4() if patch_id is None else patch_id,
        reviewer_model="grok",
        approved=False,
        objections=["The regression test does not cover the late partition."],
        round=round_number,
    )


def make_verification(
    *,
    patch_id: UUID | None = None,
    passed: bool = True,
    failures_remaining: int = 0,
) -> VerificationResult:
    return VerificationResult(
        patch_id=uuid4() if patch_id is None else patch_id,
        passed=passed,
        rows_checked=20,
        failures_remaining=failures_remaining,
        duration_ms=15,
    )


def make_approval(*, patch_id: UUID | None = None) -> ApprovalDecision:
    return ApprovalDecision(
        patch_id=uuid4() if patch_id is None else patch_id,
        decision=ApprovalOutcome.APPROVED,
        approver="U123",
        decided_at=LATER,
        comment="Ship it.",
    )


def make_postmortem(*, incident_id: UUID | None = None) -> Postmortem:
    parent = uuid4() if incident_id is None else incident_id
    return Postmortem(
        incident_id=parent,
        timeline=[
            TimelineEvent(occurred_at=NOW, actor="sentinel", summary="Detected null explosion."),
            TimelineEvent(occurred_at=LATER, actor="human", summary="Approved the patch."),
        ],
        root_cause="An upstream cast widened the null rate.",
        fix_summary="Tightened the null-rate threshold and added a regression test.",
        time_to_resolution_s=300,
        cost_usd=0.42,
    )
