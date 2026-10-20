"""Three incidents in one run stay in separate cases."""

import logging
from pathlib import Path
from uuid import uuid4

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from tests.agents.scripted_root_cause import FakeRootCauseModel
from tests.agents.support import DETECTED_AT

from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.severity_rules import floor_for
from meridian.agents.triage import run_triage
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident import Incident
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.severity import Severity, more_severe
from meridian.contracts.state import MeridianState, append_llm_calls, merge_cases
from meridian.contracts.triage import TriageResult
from meridian.graph.build import build_graph

_SPECS = (
    (FaultType.SCHEMA_DRIFT, ["trip_summary", "fleet_daily_kpi"]),
    (FaultType.DUPLICATE_KEYS, []),
    (FaultType.LATE_PARTITION, ["fleet_daily_kpi"]),
)


def _incidents() -> list[Incident]:
    return [
        Incident(
            id=uuid4(),
            detected_at=DETECTED_AT,
            dataset="vehicle_telemetry",
            partition="2026-10-19",
            contract_name="VehicleTelemetry",
            fault_type=fault,
            failing_checks=["check"],
            blast_radius=list(downstreams),
        )
        for fault, downstreams in _SPECS
    ]


def test_three_incidents_produce_independent_cases(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    incidents = _incidents()
    caplog.set_level(logging.INFO, logger="meridian.graph")

    def _collect(refs: list[object], detected_at: object, *, num_cpus: int) -> list[Incident]:
        del refs, detected_at, num_cpus
        return incidents

    monkeypatch.setattr("meridian.agents.sentinel.collect_ray", _collect)
    run_id = uuid4()
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as saver:
        graph = build_graph(
            num_cpus=1,
            model_name="fake-triage",
            chat=FakeTriageModel(),
            checkpointer=saver,
            detected_at=DETECTED_AT,
            root_model_name="fake-root-cause",
            root_chat=FakeRootCauseModel(mode="immediate"),
            evidence_root=tmp_path,
            tools=[],
        )
        finished = graph.invoke(
            MeridianState(run_id=run_id),
            {"configurable": {"thread_id": str(run_id)}},
        )
    cases = finished["cases"]
    assert set(cases) == {str(item.id) for item in incidents}
    severities = set()
    for incident in incidents:
        case = cases[str(incident.id)]
        assert case.incident == incident
        assert case.incident.severity is None
        assert case.triage is not None
        assert case.triage.incident_id == incident.id
        expected = more_severe(
            Severity.SEV4,
            floor_for(incident.fault_type, len(incident.blast_radius)),
        )
        assert case.severity is expected
        assert case.triage.final_severity is expected
        severities.add(case.severity)
    assert severities == {Severity.SEV2, Severity.SEV3, Severity.SEV4}
    triage_calls = [call for call in finished["llm_calls"] if call.node == "triage"]
    assert len(triage_calls) == 3
    assert all(call.succeeded for call in finished["llm_calls"])
    assert all(case.hypothesis is not None for case in cases.values())
    order = next(
        record.message
        for record in caplog.records
        if record.message.startswith("root cause order:")
    )
    parts = order.removeprefix("root cause order: ").split(", ")
    ranks = [part.rsplit(":", maxsplit=1)[1] for part in parts]
    assert ranks == ["SEV2", "SEV3", "SEV4"]


def test_reducer_merges_by_incident_and_appends_calls() -> None:
    first, second, _third = _incidents()
    left = {str(first.id): IncidentCase(incident=first)}
    incoming = {str(second.id): IncidentCase(incident=second)}
    merged = merge_cases(left, incoming)
    assert set(merged) == {str(first.id), str(second.id)}
    triaged = IncidentCase(
        incident=first,
        triage=TriageResult(
            incident_id=first.id,
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
    again = merge_cases(merged, {str(first.id): triaged})
    assert again[str(first.id)].triage is not None
    assert again[str(second.id)].triage is None
    assert again[str(second.id)].incident == second
    with pytest.raises(ValueError, match="cases key"):
        merge_cases({}, {"not-the-id": IncidentCase(incident=first)})
    earlier = LLMCall(
        node="triage",
        model="fake-triage",
        prompt_version="v1",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        attempt=1,
        succeeded=True,
    )
    later = earlier.model_copy(update={"input_tokens": 2})
    assert append_llm_calls([earlier], [later]) == [earlier, later]


def test_triage_branch_rejects_two_cases() -> None:
    incidents = _incidents()[:2]
    state = MeridianState(
        run_id=uuid4(),
        cases={str(item.id): IncidentCase(incident=item) for item in incidents},
    )
    with pytest.raises(ValueError, match="one incident"):
        run_triage(state, model_name="fake-triage", chat=FakeTriageModel())
