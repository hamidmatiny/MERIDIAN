"""Aborted runs end every router, and a degraded critic is retried once."""

from uuid import uuid4

import pytest
from langgraph.graph import END
from langgraph.types import Send
from tests.contracts.builders import (
    make_hypothesis,
    make_incident,
    make_patch,
    make_verdict,
    make_verification,
)

from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.remediation import RemediationKind, VerificationCheck
from meridian.contracts.severity import Severity
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageResult
from meridian.graph.build import (
    _after_critic,
    _after_pre_critic,
    _after_root,
    _after_verifier,
    _as_state,
    _fan_out,
    _guard,
    _to_pre_critic,
    _to_root,
)
from meridian.llm.errors import AbortGate, RunAborted


def _state(**overrides: object) -> MeridianState:
    values: dict[str, object] = {"run_id": uuid4()}
    values.update(overrides)
    return MeridianState(**values)  # type: ignore[arg-type]


def test_an_aborted_run_ends_every_router() -> None:
    state = _state(aborted=True, abort_reason="BILLING: credit")
    assert _fan_out(state) == END
    assert _to_root(state) == END
    assert _after_root(state) == END
    assert _to_pre_critic(state) == END
    assert _after_pre_critic(state) == END
    assert _after_critic(state) == END
    assert _after_verifier(state) == END


def test_empty_and_finished_cases_end() -> None:
    assert _to_root(_state()) == END
    assert _fan_out(_state()) == END
    incident = make_incident()
    triage = TriageResult(
        incident_id=incident.id,
        severity=Severity.SEV2,
        rationale="Floor kept this incident at SEV2.",
        affected_consumers=["trip_summary"],
        confidence=0.4,
        llm_severity=Severity.SEV4,
        floor_severity=Severity.SEV2,
        final_severity=Severity.SEV2,
        prompt_version="v1",
    )
    done = _state(cases={str(incident.id): IncidentCase(incident=incident, triage=triage)})
    assert _fan_out(done) == END
    assert _to_pre_critic(done) == END
    assert _after_root(done) == END


def test_a_blocked_pre_critic_returns_to_fix_and_a_degraded_critic_retries() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id).model_copy(
        update={"remediation_kind": RemediationKind.DATA_REPAIR}
    )
    blocked = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        fix_attempts=1,
        pre_critic_attempt=1,
        pre_critic_blocked=True,
        pre_critic_note="import os is not allowed",
    )
    sent = _after_pre_critic(_state(cases={str(incident.id): blocked}))
    assert isinstance(sent, list)
    assert isinstance(sent[0], Send)
    assert sent[0].node == "fix"
    human = blocked.model_copy(update={"needs_human": True})
    assert _after_pre_critic(_state(cases={str(incident.id): human})) == END
    verdict = make_verdict(patch_id=patch.id).model_copy(
        update={"critic_degraded": True, "approved": False, "objections": []}
    )
    degraded = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        fix_attempts=1,
        verdicts=[verdict],
    )
    retried = _after_critic(_state(cases={str(incident.id): degraded}))
    assert isinstance(retried, list)
    assert retried[0].node == "critic"


def test_guard_records_an_abort_and_rejects_a_bad_branch() -> None:
    gate = AbortGate()
    gate.trip("BILLING: credit")
    skipped = _guard(gate, _state(), lambda _state: {})
    assert skipped["aborted"] is True
    already = _guard(
        AbortGate(),
        _state(aborted=True, abort_reason="AUTH: denied"),
        lambda _state: {},
    )
    assert already["abort_reason"] == "AUTH: denied"

    def _raise(state: MeridianState) -> dict[str, object]:
        del state
        raise RunAborted("QUOTA", "limit")

    recorded = _guard(AbortGate(), _state(), _raise)
    assert recorded["aborted"] is True

    def _wrap(state: MeridianState) -> dict[str, object]:
        del state
        raise RuntimeError("graph") from RunAborted("BILLING", "credit")

    wrapped = _guard(AbortGate(), _state(), _wrap)
    assert "BILLING" in str(wrapped["abort_reason"])

    def _other(state: MeridianState) -> dict[str, object]:
        del state
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        _guard(AbortGate(), _state(), _other)

    def _flag(state: MeridianState) -> dict[str, object]:
        del state
        return {"aborted": True, "abort_reason": ""}

    flagged = _guard(AbortGate(), _state(), _flag)
    assert flagged["aborted"] is True
    parsed = _as_state({"run_id": uuid4()})
    assert isinstance(parsed, MeridianState)
    with pytest.raises(TypeError, match="MeridianState"):
        _as_state(1)


def test_a_failed_verification_returns_to_fix() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id).model_copy(
        update={"remediation_kind": RemediationKind.DATA_REPAIR}
    )
    verification = make_verification(patch_id=patch.id, passed=False, failures_remaining=1)
    verification = verification.model_copy(
        update={"checks": [VerificationCheck(name="contracts", passed=False, detail="remain")]}
    )
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        fix_attempts=1,
        verification=verification,
    )
    sent = _after_verifier(_state(cases={str(incident.id): case}))
    assert isinstance(sent, list)
    assert sent[0].node == "fix"
