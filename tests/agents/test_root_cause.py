"""Scripted root-cause loop: grounded citations, dropped hallucinations, forced answers."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from tests.agents.support import DETECTED_AT

from meridian.agents.fake_root_cause import FakeRootCauseModel
from meridian.agents.root_cause import run_root_cause
from meridian.bench.score_root_cause import score
from meridian.contracts.fault_type import FaultType
from meridian.contracts.hypothesis import Hypothesis
from meridian.contracts.incident import Incident
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.injected_fault import InjectedFault
from meridian.contracts.severity import Severity
from meridian.contracts.state import MeridianState
from meridian.injector.events import build_events, write_events
from meridian.mcp_servers.client import inprocess_tools


def _state(
    tmp_path: Path,
    *,
    fault: FaultType = FaultType.UNIT_MISMATCH,
) -> tuple[MeridianState, InjectedFault]:
    columns = ["speed_mps"]
    events, causal = build_events(
        dataset="vehicle_telemetry",
        seed=42,
        fault=fault,
        columns=columns,
        intensity=0.1,
    )
    write_events(tmp_path / "events.jsonl", events)
    assert causal is not None
    incident = Incident(
        id=uuid4(),
        detected_at=DETECTED_AT,
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=fault,
        failing_checks=["unit"],
        blast_radius=["trip_summary"],
        sample_failure_rows=[{"speed_mps": 10.0}],
    )
    case = IncidentCase(
        incident=incident,
        triage=_triage(incident.id),
    )
    label = InjectedFault(
        id=uuid4(),
        fault_type=fault,
        dataset="vehicle_telemetry",
        columns_affected=columns,
        rows_affected=4,
        intensity=0.1,
        seed=42,
        causal_event_id=causal,
    )
    state = MeridianState(
        run_id=uuid4(),
        cases={str(incident.id): case},
    )
    return state, label


def _triage(incident_id: object):
    from meridian.contracts.triage import TriageResult

    return TriageResult(
        incident_id=incident_id,
        severity=Severity.SEV2,
        rationale="Floor kept this incident at SEV2.",
        affected_consumers=["trip_summary"],
        confidence=0.4,
        llm_severity=Severity.SEV4,
        floor_severity=Severity.SEV2,
        final_severity=Severity.SEV2,
        prompt_version="v1",
    )


def _run(tmp_path: Path, mode: str, *, max_tool_calls: int = 8, tools: list[object] | None = None):
    state, label = _state(tmp_path)
    chosen = inprocess_tools(tmp_path) if tools is None else tools
    updated = run_root_cause(
        state,
        model_name="fake-root-cause",
        chat=FakeRootCauseModel(mode=mode),
        tools=chosen,  # type: ignore[arg-type]
        max_tool_calls=max_tool_calls,
        evidence_root=tmp_path,
    )
    case = updated["cases"][next(iter(state.cases))]
    return case, updated, label


def test_grounded_hypothesis_cites_the_causal_event(tmp_path: Path) -> None:
    case, updated, label = _run(tmp_path, "grounded")
    hypothesis = case.hypothesis
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.grounding_violations == 0
    assert hypothesis.causal_event_id == label.causal_event_id
    assert hypothesis.evidence[0].reference == f"event:{label.causal_event_id}"
    assert [call.tool for call in updated["tool_calls"]] == ["list_events", "get_event"]
    assert all(call.ok for call in updated["tool_calls"])
    assert all(call.node == "root_cause" for call in updated["llm_calls"])
    scored = score(hypothesis, label)
    assert scored.causal_event_correct is True
    assert scored.columns_jaccard == 1.0


def test_hallucinated_ref_is_dropped(tmp_path: Path) -> None:
    case, _updated, _label = _run(tmp_path, "hallucinate")
    hypothesis = case.hypothesis
    assert hypothesis is not None
    assert hypothesis.evidence == []
    assert hypothesis.causal_event_id is None
    assert hypothesis.grounding_violations == 2
    assert hypothesis.confidence == pytest.approx(0.3)


def test_tool_limit_forces_a_valid_hypothesis(tmp_path: Path) -> None:
    case, updated, _label = _run(tmp_path, "limit", max_tool_calls=2)
    hypothesis = case.hypothesis
    assert isinstance(hypothesis, Hypothesis)
    assert len(updated["tool_calls"]) == 2
    assert hypothesis.root_cause_summary
    assert hypothesis.mechanism
    assert hypothesis.grounding_violations == 0


def test_failures_still_return_a_hypothesis(tmp_path: Path) -> None:
    unknown, unknown_update, _label = _run(tmp_path, "unknown")
    assert unknown.hypothesis is not None
    assert unknown_update["tool_calls"][0].ok is False
    assert unknown_update["tool_calls"][0].ref_count == 0

    class _Empty(BaseModel):
        pass

    def _boom() -> str:
        raise RuntimeError("tool failed")

    boom_tool = StructuredTool.from_function(
        func=_boom,
        name="boom",
        description="fails",
        args_schema=_Empty,
    )
    boom, boom_update, _label = _run(tmp_path, "boom", tools=[boom_tool])
    assert boom.hypothesis is not None
    assert boom_update["tool_calls"][0].ok is False

    raised, raised_update, _label = _run(tmp_path, "raise-model")
    assert raised.hypothesis is not None
    assert raised_update["llm_calls"][0].succeeded is False

    bad, _bad_update, _label = _run(tmp_path, "bad-json")
    assert bad.hypothesis is not None
    assert bad.hypothesis.root_cause_summary.startswith("Tool budget exhausted")
    assert bad.hypothesis.grounding_violations == 0

    parsed, _parsed_update, _label = _run(tmp_path, "dict-draft")
    assert parsed.hypothesis is not None
    assert parsed.hypothesis.confidence == pytest.approx(0.2)

    garbage, _garbage_update, _label = _run(tmp_path, "garbage-id")
    assert garbage.hypothesis is not None
    assert garbage.hypothesis.causal_event_id is None
    assert garbage.hypothesis.grounding_violations == 1
    assert garbage.hypothesis.confidence == pytest.approx(0.55)


def test_branch_rejects_the_wrong_shape(tmp_path: Path) -> None:
    state, _label = _state(tmp_path)
    bare = state.cases[next(iter(state.cases))].model_copy(update={"triage": None})
    key = next(iter(state.cases))
    with pytest.raises(ValueError, match="after triage"):
        run_root_cause(
            state.model_copy(update={"cases": {key: bare}}),
            model_name="fake-root-cause",
            chat=FakeRootCauseModel(mode="immediate"),
            tools=[],
            max_tool_calls=2,
            evidence_root=tmp_path,
        )
    second = Incident(
        id=uuid4(),
        detected_at=DETECTED_AT,
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=FaultType.UNIT_MISMATCH,
        failing_checks=["unit"],
    )
    two = {
        key: state.cases[key],
        str(second.id): IncidentCase(incident=second, triage=_triage(second.id)),
    }
    with pytest.raises(ValueError, match="one incident"):
        run_root_cause(
            state.model_copy(update={"cases": two}),
            model_name="fake-root-cause",
            chat=FakeRootCauseModel(mode="immediate"),
            tools=[],
            max_tool_calls=2,
            evidence_root=tmp_path,
        )


def test_score_reports_a_miss_and_an_empty_overlap() -> None:
    label = InjectedFault(
        id=uuid4(),
        fault_type=FaultType.UNIT_MISMATCH,
        dataset="vehicle_telemetry",
        columns_affected=["speed_mps"],
        rows_affected=4,
        intensity=0.1,
        seed=42,
        causal_event_id=uuid4(),
    )
    missed = Hypothesis(
        id=uuid4(),
        incident_id=uuid4(),
        root_cause_summary="wrong event",
        evidence=[],
        confidence=0.1,
        causal_event_id=None,
        suspect_columns=["lat"],
        mechanism="not the cause",
    )
    scored = score(missed, label)
    assert scored.causal_event_correct is False
    assert scored.columns_jaccard == 0.0
    empty = missed.model_copy(update={"suspect_columns": []})
    blank = label.model_copy(update={"columns_affected": []})
    assert score(empty, blank).columns_jaccard == 1.0


def test_partial_column_overlap() -> None:
    label = InjectedFault(
        id=uuid4(),
        fault_type=FaultType.SCHEMA_DRIFT,
        dataset="vehicle_telemetry",
        columns_affected=["speed_mps", "lat"],
        rows_affected=1,
        intensity=0.3,
        seed=1,
        causal_event_id=uuid4(),
    )
    hypothesis = Hypothesis(
        id=uuid4(),
        incident_id=uuid4(),
        root_cause_summary="partial",
        confidence=0.4,
        causal_event_id=label.causal_event_id,
        suspect_columns=["speed_mps", "lon"],
        mechanism="rename",
    )
    scored = score(hypothesis, label)
    assert scored.causal_event_correct is True
    assert scored.columns_jaccard == pytest.approx(1 / 3)
    print(json.dumps(scored.model_dump()))
