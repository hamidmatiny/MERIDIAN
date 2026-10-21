"""Triage floor, retries, and a skipped live smoke."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from tests.agents.support import DETECTED_AT, write_partition

from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.severity_rules import floor_for
from meridian.agents.triage import run_triage
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.contracts.severity import Severity
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageAssessment, TriageResult
from meridian.pipeline.parquet import read_parquet
from meridian.prompts import load_prompt
from meridian.quality.evaluate import evaluate


def _incident(tmp_path: Path) -> MeridianState:
    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=FaultType.SCHEMA_DRIFT,
        intensity=0.3,
        seed=11,
        rows=60,
        name="one",
    )
    profile = ReferenceProfile.model_validate_json(
        Path(ref.reference_profile_path).read_text(encoding="utf-8")
    )
    incidents = evaluate(
        ref.dataset,
        read_parquet(Path(ref.parquet_path)),
        ref.metadata,
        profile,
        detected_at=DETECTED_AT,
    )
    chosen = [item for item in incidents if item.fault_type is FaultType.SCHEMA_DRIFT]
    assert len(chosen) == 1
    incident = chosen[0]
    assert incident.severity is None
    return MeridianState(
        run_id=uuid4(),
        partitions=[ref],
        cases={str(incident.id): IncidentCase(incident=incident)},
    )


def _only_case(updated: dict[str, object]) -> IncidentCase:
    cases = updated["cases"]
    assert isinstance(cases, dict)
    assert len(cases) == 1
    case = next(iter(cases.values()))
    assert isinstance(case, IncidentCase)
    assert case.triage is not None
    return case


def test_llm_cannot_lower_a_sev2_floor(tmp_path: Path) -> None:
    state = _incident(tmp_path)
    incident = next(iter(state.cases.values())).incident
    floor = floor_for(incident.fault_type, len(incident.blast_radius))
    assert floor.value == "SEV2"
    updated = run_triage(state, model_name="fake-triage", chat=FakeTriageModel())
    case = _only_case(updated)
    result = case.triage
    assert result is not None
    assert result.llm_severity is not None
    assert result.llm_severity.value == "SEV4"
    assert result.floor_severity is floor
    assert result.final_severity.value == "SEV2"
    assert result.severity is result.final_severity
    assert result.triage_degraded is False
    assert result.prompt_version == "v1"
    assert updated["llm_calls"][0].succeeded is True
    assert updated["llm_calls"][0].input_tokens == 12
    assert case.incident.severity is None
    assert case.severity is result.final_severity


def test_two_invalid_replies_degrade_to_the_floor(tmp_path: Path) -> None:
    state = _incident(tmp_path)
    updated = run_triage(
        state,
        model_name="fake-triage",
        chat=FakeTriageModel(failures=2),
    )
    result = _only_case(updated).triage
    assert result is not None
    assert result.triage_degraded is True
    assert result.llm_severity is None
    assert result.final_severity is result.floor_severity
    assert [call.succeeded for call in updated["llm_calls"]] == [False, False]
    assert [call.attempt for call in updated["llm_calls"]] == [1, 2]


def test_missing_prompt_is_rejected() -> None:
    with pytest.raises(FileNotFoundError, match="missing prompt"):
        load_prompt("triage", "v9")


@pytest.mark.live
def test_live_triage_smoke(tmp_path: Path) -> None:
    if os.environ.get("MERIDIAN_LIVE") != "1":
        pytest.skip("set MERIDIAN_LIVE=1 to call the configured model")
    from meridian.config import get_settings
    from meridian.llm.router import build_node_chat

    settings = get_settings()
    if settings.triage_provider in {"anthropic", "xai"} and not settings.allow_paid:
        pytest.skip("paid provider requires MERIDIAN_ALLOW_PAID=1")
    chat, model_name = build_node_chat(settings, "triage")
    from meridian.llm.capabilities import arm_node

    arm_node(settings, "triage", chat)
    updated = run_triage(
        _incident(tmp_path),
        model_name=model_name,
        chat=chat,
    )
    case = _only_case(updated)
    assert case.triage is not None
    assert case.severity is case.triage.final_severity
    assert case.incident.severity is None
    assert case.triage.llm_severity is not None
    assert case.triage.triage_degraded is False
    assert any(call.succeeded for call in updated["llm_calls"])
    for call in updated["llm_calls"]:
        usd = "0.00" if call.provider not in {"anthropic", "xai"} else "paid"
        print(
            f"provider={call.provider} model={call.model} "
            f"input_tokens={call.input_tokens} output_tokens={call.output_tokens} usd={usd}"
        )
    print(
        json.dumps(
            {
                "severity": case.severity.value if case.severity is not None else None,
                "rationale": case.triage.rationale,
                "llm_severity": (
                    None if case.triage.llm_severity is None else case.triage.llm_severity.value
                ),
                "floor_severity": case.triage.floor_severity.value,
                "final_severity": case.triage.final_severity.value,
                "llm_calls": [call.model_dump(mode="json") for call in updated["llm_calls"]],
            },
            indent=2,
        )
    )


class _WrongIncident:
    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: object,
    ) -> "_WrongIncident":
        del schema, include_raw, kwargs
        return self

    def invoke(self, messages: object) -> dict[str, object]:
        del messages
        raw = AIMessage(
            content="wrong",
            usage_metadata={"input_tokens": 1, "output_tokens": True},
        )
        parsed = TriageAssessment(
            incident_id=uuid4(),
            severity=Severity.SEV1,
            rationale="This assessment names a different incident.",
            affected_consumers=[],
            confidence=0.1,
        )
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class _Raises:
    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: object,
    ) -> "_Raises":
        del schema, include_raw, kwargs
        return self

    def invoke(self, messages: object) -> dict[str, object]:
        del messages
        raise RuntimeError("model unavailable")


def test_mismatched_incident_id_degrades(tmp_path: Path) -> None:
    updated = run_triage(_incident(tmp_path), model_name="fake-triage", chat=_WrongIncident())
    result = _only_case(updated).triage
    assert result is not None
    assert result.triage_degraded is True
    assert updated["llm_calls"][0].succeeded is False
    assert updated["llm_calls"][0].output_tokens == 0


def test_model_exception_degrades_to_the_floor(tmp_path: Path) -> None:
    updated = run_triage(_incident(tmp_path), model_name="fake-triage", chat=_Raises())
    result = _only_case(updated).triage
    assert result is not None
    assert result.triage_degraded is True
    assert updated["llm_calls"][0].input_tokens == 0


def test_fake_model_ignores_a_missing_downstream() -> None:
    model = FakeTriageModel()
    missing = model.next_text(
        [HumanMessage(content="Incident-Id: 00000000-0000-0000-0000-000000000001\n")]
    )
    blank = model.next_text(
        [HumanMessage(content="Incident-Id: 00000000-0000-0000-0000-000000000001\nDownstream: -")]
    )
    assert json.loads(missing)["affected_consumers"] == []
    assert json.loads(blank)["affected_consumers"] == []


def _result(**overrides: object) -> TriageResult:
    payload: dict[str, object] = {
        "incident_id": uuid4(),
        "severity": Severity.SEV2,
        "rationale": "Recorded for the validator checks.",
        "affected_consumers": ["trip_summary"],
        "confidence": 0.5,
        "llm_severity": Severity.SEV4,
        "floor_severity": Severity.SEV2,
        "final_severity": Severity.SEV2,
        "prompt_version": "v1",
        "triage_degraded": False,
    }
    payload.update(overrides)
    return TriageResult.model_validate(payload)


def test_triage_result_rejects_inconsistent_severity() -> None:
    with pytest.raises(ValueError, match="severity must equal final_severity"):
        _result(severity=Severity.SEV3)
    with pytest.raises(ValueError, match="degraded triage keeps the floor"):
        _result(
            severity=Severity.SEV2,
            final_severity=Severity.SEV2,
            floor_severity=Severity.SEV3,
            llm_severity=None,
            triage_degraded=True,
        )
    with pytest.raises(ValueError, match="a completed triage records llm_severity"):
        _result(llm_severity=None)
    with pytest.raises(ValueError, match="more severe"):
        _result(severity=Severity.SEV4, final_severity=Severity.SEV4, llm_severity=Severity.SEV4)
