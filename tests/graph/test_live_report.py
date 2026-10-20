"""The live stderr report includes provider, verdict, and score lines."""

from uuid import uuid4

import pandas as pd
from tests.contracts.builders import (
    make_hypothesis,
    make_incident,
    make_patch,
    make_verdict,
    make_verification,
)

from meridian.contracts.dry_run import DryRunReport
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.injected_fault import InjectedFault
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.remediation import VerificationCheck
from meridian.contracts.severity import Severity
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageResult
from meridian.run import _report


def test_report_lists_nodes_verdicts_and_scores() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    triage = TriageResult(
        incident_id=incident.id,
        severity=Severity.SEV2,
        rationale="Floor kept this incident at SEV2.",
        affected_consumers=["trip_summary"],
        confidence=0.4,
        llm_severity=Severity.SEV2,
        floor_severity=Severity.SEV2,
        final_severity=Severity.SEV2,
        prompt_version="v1",
    )
    verification = make_verification(patch_id=patch.id).model_copy(
        update={"checks": [VerificationCheck(name="contracts", passed=True, detail="clear")]}
    )
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        triage=triage,
        patch=patch,
        dry_run=DryRunReport(
            rows_sampled=10,
            rows_changed=2,
            rows_changed_outside=0,
            over_broad=False,
            contracts_after=1,
            psi_after="under threshold",
        ),
        verdicts=[make_verdict(patch_id=patch.id)],
        verification=verification,
    )
    label = InjectedFault(
        id=uuid4(),
        fault_type=FaultType.NULL_EXPLOSION,
        dataset="orders",
        columns_affected=["amount"],
        rows_affected=1,
        intensity=0.1,
        seed=1,
        causal_event_id=uuid4(),
    )
    state = MeridianState(
        run_id=uuid4(),
        aborted=True,
        abort_reason="QUOTA: limit",
        cases={str(incident.id): case},
        llm_calls=[
            LLMCall(
                node="triage",
                model="qwen3.5:4b",
                provider="ollama",
                prompt_version="v1",
                input_tokens=10,
                output_tokens=4,
                prompt_tokens_before=12,
                prompt_tokens_after=8,
                latency_ms=5,
                attempt=1,
                succeeded=True,
            ),
            LLMCall(
                node="critic",
                model="grok-4.7",
                provider="xai",
                prompt_version="v1",
                input_tokens=3,
                output_tokens=2,
                latency_ms=5,
                attempt=1,
                succeeded=True,
                fallback_used=True,
            ),
        ],
    )
    text = _report(state, wall_s=1.25, clean=pd.DataFrame({"amount": [1.0]}), label=label)
    assert "aborted=True" in text
    assert "provider=ollama" in text
    assert "usd=0.00" in text
    assert "usd=paid" in text
    assert "llm_severity=SEV2" in text
    assert "dry_run" in text
    assert "critic_degraded=False" in text
    assert "contracts:pass" in text
    assert "score_root_cause" in text
    assert "score_fix" in text
    assert "wall_s=1.25" in text
