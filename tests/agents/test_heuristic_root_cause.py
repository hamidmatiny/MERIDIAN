"""The CLI fake cites tool output. It does not read a ground-truth label."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage, ToolMessage
from tests.agents.support import DETECTED_AT

from meridian.agents.fake_root_cause import HeuristicRootCauseModel
from meridian.agents.root_cause import run_root_cause
from meridian.contracts.change_event import ChangeEvent, EventKind
from meridian.contracts.fault_type import FaultType
from meridian.contracts.hypothesis import Hypothesis
from meridian.contracts.incident import Incident
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.severity import Severity
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageResult
from meridian.injector.events import write_events
from meridian.mcp_servers.client import inprocess_tools

_BANNED = ("Causal event:", "InjectedFault", "injected_fault", "events.jsonl")


def test_cli_fake_source_has_no_ground_truth_hooks() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/meridian/agents/fake_root_cause.py").read_text(encoding="utf-8")
    run_source = (root / "src/meridian/run.py").read_text(encoding="utf-8")
    for banned in _BANNED:
        assert banned not in source
    assert "HeuristicRootCauseModel" in run_source
    assert "FakeRootCauseModel" not in run_source
    assert "load_stdio_tools" not in run_source


def test_heuristic_picks_the_newest_event_on_the_dataset(tmp_path: Path) -> None:
    older_id = uuid4()
    newer_id = uuid4()
    other_id = uuid4()
    write_events(
        tmp_path / "events.jsonl",
        [
            ChangeEvent(
                id=older_id,
                ts=datetime(2026, 10, 16, 12, 0, tzinfo=UTC),
                kind=EventKind.SCHEMA_CHANGE,
                dataset="vehicle_telemetry",
                author="pipeline",
                summary="Causal event: renamed speed_mps on vehicle_telemetry",
                diff_excerpt="- speed_mps\n+ speed_mps_renamed",
            ),
            ChangeEvent(
                id=newer_id,
                ts=datetime(2026, 10, 19, 18, 0, tzinfo=UTC),
                kind=EventKind.DEPLOY,
                dataset="vehicle_telemetry",
                author="pipeline",
                summary="Distractor: routine DEPLOY on vehicle_telemetry",
                diff_excerpt="",
            ),
            ChangeEvent(
                id=other_id,
                ts=datetime(2026, 10, 19, 19, 0, tzinfo=UTC),
                kind=EventKind.DEPLOY,
                dataset="trip_summary",
                author="pipeline",
                summary="Newer deploy on another dataset",
                diff_excerpt="",
            ),
        ],
    )
    case, calls = _run(tmp_path)
    hypothesis = case.hypothesis
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.causal_event_id == newer_id
    assert hypothesis.causal_event_id != older_id
    assert hypothesis.evidence[0].reference == f"event:{newer_id}"
    assert hypothesis.suspect_columns == []
    assert hypothesis.grounding_violations == 0
    assert [call.tool for call in calls] == ["list_events", "get_event"]
    assert all(call.ok for call in calls)


def test_heuristic_ignores_payloads_that_are_not_events() -> None:
    model = HeuristicRootCauseModel()
    messages = [
        HumanMessage(content=[{"type": "text", "text": "plain"}]),
        ToolMessage(content="not-json", tool_call_id="bad"),
        ToolMessage(content=[{"x": 1}], tool_call_id="blocks"),
    ]
    answer = model.bind_tools([]).invoke(messages)
    assert answer.tool_calls == []
    draft = model.with_structured_output(object).invoke(messages)["parsed"]
    assert draft.causal_event_id is None
    assert draft.evidence == []


def test_heuristic_answers_when_the_dataset_has_no_events(tmp_path: Path) -> None:
    write_events(tmp_path / "events.jsonl", [])
    case, calls = _run(tmp_path)
    hypothesis = case.hypothesis
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.causal_event_id is None
    assert hypothesis.evidence == []
    assert hypothesis.grounding_violations == 0
    assert [call.tool for call in calls] == ["list_events"]


def _run(tmp_path: Path):
    incident = Incident(
        id=uuid4(),
        detected_at=DETECTED_AT,
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=FaultType.SCHEMA_DRIFT,
        failing_checks=["columns"],
        blast_radius=["trip_summary"],
    )
    state = MeridianState(
        run_id=uuid4(),
        cases={
            str(incident.id): IncidentCase(
                incident=incident,
                triage=TriageResult(
                    incident_id=incident.id,
                    severity=Severity.SEV2,
                    rationale="Floor kept this incident at SEV2.",
                    affected_consumers=["trip_summary"],
                    confidence=0.4,
                    llm_severity=Severity.SEV4,
                    floor_severity=Severity.SEV2,
                    final_severity=Severity.SEV2,
                    prompt_version="v1",
                ),
            )
        },
    )
    updated = run_root_cause(
        state,
        model_name="heuristic-root-cause",
        chat=HeuristicRootCauseModel(),
        tools=inprocess_tools(tmp_path),
        max_tool_calls=8,
        evidence_root=tmp_path,
    )
    case = updated["cases"][str(incident.id)]
    return case, updated["tool_calls"]
