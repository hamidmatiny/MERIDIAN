"""A billing error on root cause stops every later model call."""

from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from tests.agents.support import DETECTED_AT

from meridian.agents.fake_llm import FakeTriageModel
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident import Incident
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph


class _Credit:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: object, **kwargs: object) -> "_Credit":
        del tools, kwargs
        return self

    def with_structured_output(self, schema: object, **kwargs: object) -> "_Credit":
        del schema, kwargs
        return self

    def invoke(self, messages: object, **kwargs: object) -> object:
        del messages, kwargs
        self.calls += 1
        error = RuntimeError("Your credit balance is too low")
        error.status_code = 400  # type: ignore[attr-defined]
        raise error


class _Counting:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: object, **kwargs: object) -> "_Counting":
        del tools, kwargs
        return self

    def with_structured_output(self, schema: object, **kwargs: object) -> "_Counting":
        del schema, kwargs
        return self

    def invoke(self, messages: object, **kwargs: object) -> object:
        del messages, kwargs
        self.calls += 1
        raise AssertionError("called after a billing error")


def test_credit_error_on_root_cause_makes_no_later_calls(tmp_path: Path, monkeypatch) -> None:
    incident = Incident(
        id=uuid4(),
        detected_at=DETECTED_AT,
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=FaultType.UNIT_MISMATCH,
        failing_checks=["speed_mps"],
    )
    monkeypatch.setattr(
        "meridian.agents.sentinel.collect_ray",
        lambda refs, detected_at, *, num_cpus: [incident],
    )
    root = _Credit()
    fix = _Counting()
    critic = _Counting()
    run_id = uuid4()
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as saver:
        graph = build_graph(
            num_cpus=1,
            model_name="fake-triage",
            chat=FakeTriageModel(),
            checkpointer=saver,
            detected_at=DETECTED_AT,
            root_model_name="fake-root",
            root_chat=root,
            evidence_root=tmp_path,
            tools=[],
            fix_model_name="fake-fix",
            fix_chat=fix,
            critic_model_name="fake-critic",
            critic_chat=critic,
        )
        finished = graph.invoke(
            MeridianState(run_id=run_id),
            {"configurable": {"thread_id": str(run_id)}},
        )
        snapshot = graph.get_state({"configurable": {"thread_id": str(run_id)}})
    state = MeridianState.model_validate(finished)
    assert state.aborted is True
    assert "BILLING" in state.abort_reason
    assert root.calls == 1
    assert fix.calls == 0
    assert critic.calls == 0
    assert all(call.node != "fix" and call.node != "critic" for call in state.llm_calls)
    case = next(iter(state.cases.values()))
    assert case.hypothesis is None
    assert snapshot.values["aborted"] is True
