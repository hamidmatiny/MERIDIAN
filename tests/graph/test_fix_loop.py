"""Scripted Fix and Critic paths, including the verifier caps."""

from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from tests.agents.scripted_critic import ScriptedCriticModel
from tests.agents.scripted_fix import ScriptedFixModel
from tests.agents.scripted_root_cause import FakeRootCauseModel
from tests.agents.support import write_partition
from tests.contracts.builders import make_hypothesis, make_incident, make_patch

from meridian.agents.critic import run_critic
from meridian.agents.fake_fix import _CHECK_REPAIR, _REPAIR
from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.fix import PatchDraft, run_fix
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph

_BAD = "import os\ndef repair(df):\n    return df\n"
_CHECK = "def check(df):\n    return None\n"


def _draft(kind: str, source: str, test: str = _CHECK_REPAIR) -> PatchDraft:
    return PatchDraft(remediation_kind=kind, module_source=source, regression_test=test)


def _repair() -> PatchDraft:
    return _draft(RemediationKind.DATA_REPAIR.value, _REPAIR)


def _run(
    tmp_path: Path,
    drafts: list[PatchDraft],
    approved: list[bool],
    *,
    monkeypatch: object | None = None,
    recorded: list[object] | None = None,
) -> MeridianState:
    from meridian.contracts.fault_type import FaultType

    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=FaultType.SCHEMA_DRIFT,
        intensity=0.3,
        seed=3,
        rows=400,
        name="drift",
    )
    run_id = uuid4()
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as saver:
        graph = build_graph(
            num_cpus=1,
            model_name="fake-triage",
            chat=FakeTriageModel(),
            checkpointer=saver,
            root_model_name="fake-root-cause",
            root_chat=FakeRootCauseModel(mode="immediate"),
            evidence_root=tmp_path,
            tools=[],
            fix_chat=ScriptedFixModel(drafts),
            critic_chat=ScriptedCriticModel(approved),
            sandbox_timeout_s=30,
        )
        if monkeypatch is not None and recorded is not None:
            _capture(monkeypatch, recorded)
        finished = graph.invoke(
            MeridianState(run_id=run_id, partitions=[ref]),
            {"configurable": {"thread_id": str(run_id)}, "recursion_limit": 40},
        )
    return MeridianState.model_validate(finished)


def _capture(monkeypatch: object, recorded: list[object]) -> None:
    def wrapped(state: MeridianState, **kwargs: object) -> dict[str, object]:
        from meridian.agents.verifier import run_verifier

        out = run_verifier(state, **kwargs)
        recorded.append(next(iter(out["cases"].values())).verification)
        return out

    monkeypatch.setattr("meridian.graph.build.run_verifier", wrapped)  # type: ignore[attr-defined]


def _case(state: MeridianState) -> IncidentCase:
    assert len(state.cases) == 1
    return next(iter(state.cases.values()))


def test_approve_path(tmp_path: Path) -> None:
    state = _run(tmp_path, [_repair()], [True])
    case = _case(state)
    assert case.patch is not None
    assert case.patch.remediation_kind is RemediationKind.DATA_REPAIR
    assert case.verdicts[-1].approved
    assert case.verification is not None
    assert case.verification.passed
    assert case.needs_human is False


def test_reject_then_revise_then_approve(tmp_path: Path) -> None:
    state = _run(tmp_path, [_repair(), _repair()], [False, True])
    case = _case(state)
    assert case.critic_rounds == 2
    assert [item.approved for item in case.verdicts] == [False, True]
    assert case.verification is not None and case.verification.passed
    assert case.needs_human is False


def test_three_rejections_need_a_human(tmp_path: Path) -> None:
    state = _run(tmp_path, [_repair(), _repair(), _repair()], [False, False, False])
    case = _case(state)
    assert case.critic_rounds == 3
    assert case.fix_attempts == 3
    assert case.needs_human is True
    assert case.verification is None
    assert case.critic_rejection_rate == 1.0


_NOOP = "def repair(df):\n    return df\n"


def test_static_gate_failure_returns_to_fix_without_a_critic_round(tmp_path: Path) -> None:
    state = _run(
        tmp_path,
        [_draft(RemediationKind.DATA_REPAIR.value, _BAD, _CHECK), _repair()],
        [True],
    )
    case = _case(state)
    assert case.fix_attempts == 2
    assert case.critic_rounds == 1
    assert case.verification is not None and case.verification.passed
    assert case.needs_human is False


def test_static_gate_failure_at_the_fix_cap_needs_a_human(tmp_path: Path) -> None:
    bad = _draft(RemediationKind.DATA_REPAIR.value, _BAD, _CHECK)
    state = _run(tmp_path, [bad, bad, bad], [True])
    case = _case(state)
    assert case.fix_attempts == 3
    assert case.critic_rounds == 0
    assert case.pre_critic_blocked is True
    assert "import os" in case.pre_critic_note
    assert case.verification is None
    assert case.needs_human is True


def test_verifier_failure_then_refix_passes(tmp_path: Path, monkeypatch) -> None:
    recorded: list[object] = []
    noop = _draft(RemediationKind.DATA_REPAIR.value, _NOOP, _CHECK)
    state = _run(
        tmp_path,
        [noop, _repair()],
        [True, True],
        monkeypatch=monkeypatch,
        recorded=recorded,
    )
    case = _case(state)
    assert recorded[0].checks[0].name == "contracts"  # type: ignore[attr-defined]
    assert recorded[0].passed is False  # type: ignore[attr-defined]
    assert case.verification is not None and case.verification.passed
    assert case.fix_attempts == 2
    assert case.needs_human is False


def test_verifier_failure_at_the_limit_needs_a_human(tmp_path: Path) -> None:
    bad = _draft(RemediationKind.DATA_REPAIR.value, _NOOP, _CHECK)
    state = _run(tmp_path, [bad, bad, bad], [True, True, True])
    case = _case(state)
    assert case.fix_attempts == 3
    assert case.needs_human is True
    assert case.verification is not None
    assert case.verification.passed is False
    assert case.verification.checks[0].name == "contracts"


def test_escalate_path(tmp_path: Path) -> None:
    draft = _draft(
        RemediationKind.ESCALATE.value,
        "def repair(df):\n    return df\n",
        _CHECK,
    )
    state = _run(tmp_path, [draft], [True])
    case = _case(state)
    assert case.patch is not None
    assert case.patch.remediation_kind is RemediationKind.ESCALATE
    assert case.needs_human is True
    assert case.verification is not None
    assert case.verification.passed is False
    assert case.verification.checks[0].name == "escalated"


def test_fix_tool_call_is_recorded() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    case = IncidentCase(incident=incident, hypothesis=hypothesis)

    class _Tool:
        name = "list_events"

        def invoke(self, args: dict[str, object]) -> str:
            del args
            return '{"ref": "events:orders"}'

    state = run_fix(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        model_name="scripted-fix",
        chat=ScriptedFixModel([_repair()], tool_name="list_events"),
        tools=[_Tool()],  # type: ignore[list-item]
        max_tool_calls=4,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    updated = state["cases"][str(incident.id)]
    assert updated.patch is not None
    assert state["tool_calls"][0].ok is True
    assert state["tool_calls"][0].ref_count == 1


def test_fix_parse_failure_escalates() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    case = IncidentCase(incident=incident, hypothesis=hypothesis)

    class _Broken:
        def bind_tools(self, tools: object, **kwargs: object) -> "_Broken":
            del tools, kwargs
            return self

        def with_structured_output(self, schema: object, **kwargs: object) -> "_Broken":
            del schema, kwargs
            return self

        def invoke(self, messages: object, **kwargs: object) -> dict[str, object]:
            del messages, kwargs
            return {"raw": None, "parsed": None, "parsing_error": ValueError("no")}

    state = run_fix(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        model_name="broken",
        chat=_Broken(),
        tools=[],
        max_tool_calls=2,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    patch = state["cases"][str(incident.id)].patch
    assert patch is not None
    assert patch.remediation_kind is RemediationKind.ESCALATE
    assert state["llm_calls"][-1].succeeded is False


def test_critic_parse_failure_rejects() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    case = IncidentCase(incident=incident, hypothesis=hypothesis, patch=patch)

    class _Broken:
        def with_structured_output(self, schema: object, **kwargs: object) -> "_Broken":
            del schema, kwargs
            return self

        def invoke(self, messages: object, **kwargs: object) -> object:
            del messages, kwargs
            raise RuntimeError("down")

    state = run_critic(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        model_name="broken-critic",
        chat=_Broken(),
    )
    updated = state["cases"][str(incident.id)]
    assert updated.verdicts[-1].approved is False
    assert updated.verdicts[-1].critic_degraded is True
    assert updated.verdicts[-1].objections == []
    assert state["llm_calls"][0].succeeded is False
