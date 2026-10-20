"""No generated sequence of verdicts and verifications can pass the loop caps."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.contracts.builders import make_hypothesis, make_incident

from meridian.contracts.incident_case import MAX_CRITIC_ROUNDS, MAX_FIX_ATTEMPTS, IncidentCase
from meridian.contracts.patch import Patch
from meridian.contracts.remediation import Objection, ObjectionKind, RemediationKind
from meridian.contracts.verdict import Verdict
from meridian.contracts.verification import VerificationResult
from meridian.graph.limits import (
    begin_fix,
    finish_fix,
    kind_or_escalate,
    patch_id_for,
    record_verdict,
    record_verification,
    rejection_rate,
    route_after_critic,
    route_after_root,
    route_after_verifier,
)

_ACTIONS = st.sampled_from(["reject", "approve", "fail", "pass", "escalate"])


def _start() -> IncidentCase:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    return IncidentCase(incident=incident, hypothesis=hypothesis)


def _patch(case: IncidentCase) -> Patch:
    assert case.hypothesis is not None
    return Patch(
        id=patch_id_for(case.incident.id, "v1"),
        incident_id=case.incident.id,
        hypothesis_id=case.hypothesis.id,
        remediation_kind=RemediationKind.DATA_REPAIR,
        diff="--- /dev/null\n+++ b/remediations/x.py\n@@ -0,0 +1 @@\n+return df\n",
        regression_test="def check(df):\n    return None\n",
        files_touched=[f"remediations/{case.incident.id}.py"],
        author_model="script",
    )


@given(st.lists(_ACTIONS, max_size=8))
@settings(max_examples=40, deadline=None)
def test_no_sequence_exceeds_the_loop_limits(actions: list[str]) -> None:
    case = _start()
    phase = route_after_root(case)
    for action in actions:
        assert case.fix_attempts <= MAX_FIX_ATTEMPTS
        assert case.critic_rounds <= MAX_CRITIC_ROUNDS
        if phase == "end" or case.needs_human:
            break
        if phase == "fix":
            case = finish_fix(begin_fix(case), _patch(case))
            phase = "critic"
            continue
        if phase == "critic":
            assert case.patch is not None
            approved = action == "approve"
            case = record_verdict(
                case,
                Verdict(
                    patch_id=case.patch.id,
                    reviewer_model="grok",
                    approved=approved,
                    objections=[]
                    if approved
                    else [Objection(kind=ObjectionKind.OTHER, message="revise")],
                    round=case.critic_rounds + 1,
                ),
            )
            phase = route_after_critic(case)
            continue
        assert case.patch is not None
        passed = action == "pass"
        case = record_verification(
            case,
            VerificationResult(
                patch_id=case.patch.id,
                remediation_kind=case.patch.remediation_kind,
                passed=passed,
                rows_checked=4,
                failures_remaining=0 if passed else 1,
                duration_ms=1,
            ),
            escalate=action == "escalate",
        )
        phase = route_after_verifier(case)
    assert case.fix_attempts <= MAX_FIX_ATTEMPTS
    assert case.critic_rounds <= MAX_CRITIC_ROUNDS
    if case.fix_attempts == MAX_FIX_ATTEMPTS:
        with pytest.raises(ValueError, match="fix attempt cap"):
            begin_fix(case)
    if case.critic_rounds == MAX_CRITIC_ROUNDS and case.patch is not None:
        with pytest.raises(ValueError, match="critic round cap"):
            record_verdict(
                case,
                Verdict(
                    patch_id=case.patch.id,
                    reviewer_model="grok",
                    approved=False,
                    objections=[Objection(kind=ObjectionKind.OTHER, message="again")],
                    round=1,
                ),
            )


def test_routes_stop_at_the_caps() -> None:
    case = _start()
    assert route_after_root(case) == "fix"
    case = finish_fix(begin_fix(case), _patch(case))
    assert case.patch is not None
    case = record_verdict(
        case,
        Verdict(
            patch_id=case.patch.id,
            reviewer_model="grok",
            approved=True,
            round=1,
        ),
    )
    assert route_after_critic(case) == "verifier"
    case = record_verification(
        case,
        VerificationResult(
            patch_id=case.patch.id,
            passed=True,
            rows_checked=1,
            failures_remaining=0,
            duration_ms=1,
        ),
        escalate=False,
    )
    assert route_after_verifier(case) == "end"
    assert kind_or_escalate("NOPE") is RemediationKind.ESCALATE
    assert rejection_rate([]) == 0.0
