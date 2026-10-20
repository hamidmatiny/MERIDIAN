"""Checkpoint resume runs triage once and does not repeat Sentinel."""

from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from tests.agents.support import DETECTED_AT, write_partition

import meridian.graph.build as build_mod
from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.fake_root_cause import FakeRootCauseModel
from meridian.contracts.fault_type import FaultType
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph


def test_clean_partition_stops_after_sentinel(tmp_path: Path, monkeypatch) -> None:
    ref = write_partition(
        tmp_path,
        dataset="fleet_daily_kpi",
        fault=None,
        intensity=0,
        seed=7,
        rows=2_000,
        name="clean",
    )
    triage_calls = {"n": 0}
    real_triage = build_mod.run_triage

    def _triage(*args, **kwargs):
        triage_calls["n"] += 1
        return real_triage(*args, **kwargs)

    monkeypatch.setattr(build_mod, "run_triage", _triage)
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
        result = MeridianState.model_validate(
            graph.invoke(
                MeridianState(run_id=uuid4(), partitions=[ref]),
                {"configurable": {"thread_id": "clean"}},
            )
        )
    assert result.cases == {}
    assert result.llm_calls == []
    assert triage_calls["n"] == 0


def test_resume_after_sentinel_does_not_rerun_it(tmp_path: Path, monkeypatch) -> None:
    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=FaultType.SCHEMA_DRIFT,
        intensity=0.3,
        seed=42,
        rows=80,
        name="drift",
    )
    sentinel_calls = {"n": 0}
    real_sentinel = build_mod.run_sentinel

    def _sentinel(*args, **kwargs):
        sentinel_calls["n"] += 1
        return real_sentinel(*args, **kwargs)

    monkeypatch.setattr(build_mod, "run_sentinel", _sentinel)
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as saver:
        graph = build_graph(
            num_cpus=1,
            model_name="fake-triage",
            chat=FakeTriageModel(),
            checkpointer=saver,
            detected_at=DETECTED_AT,
            interrupt_after=["sentinel"],
            root_model_name="fake-root-cause",
            root_chat=FakeRootCauseModel(mode="immediate"),
            evidence_root=tmp_path,
            tools=[],
        )
        config = {"configurable": {"thread_id": "resume-me"}}
        paused = MeridianState.model_validate(
            graph.invoke(MeridianState(run_id=uuid4(), partitions=[ref]), config)
        )
        assert paused.cases
        assert all(case.triage is None for case in paused.cases.values())
        assert all(case.severity is None for case in paused.cases.values())
        assert sentinel_calls["n"] == 1
        finished = MeridianState.model_validate(graph.invoke(None, config))
    assert sentinel_calls["n"] == 1
    schema = [
        case
        for case in finished.cases.values()
        if case.incident.fault_type is FaultType.SCHEMA_DRIFT
    ]
    assert schema
    assert schema[0].triage is not None
    assert schema[0].incident.severity is None
    assert schema[0].triage.llm_severity is not None
    assert schema[0].triage.llm_severity.value == "SEV4"
    assert schema[0].severity is schema[0].triage.final_severity
    assert schema[0].severity is not None
    assert schema[0].severity.value == "SEV2"
