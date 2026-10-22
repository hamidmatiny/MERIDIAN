"""Verifier pass interrupts. Resume applies only an approval. No network."""

from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from tests.agents.scripted_critic import ScriptedCriticModel
from tests.agents.scripted_fix import ScriptedFixModel
from tests.agents.scripted_root_cause import FakeRootCauseModel
from tests.agents.support import write_partition

from meridian.agents.fake_fix import _CHECK_REPAIR, _REPAIR
from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.fix import PatchDraft
from meridian.contracts.approval import ApprovalOutcome
from meridian.contracts.fault_type import FaultType
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph
from meridian.slack.blocks import button_kinds
from meridian.slack.gate import ApprovalPorts
from meridian.slack.gateway import RecordingGateway
from meridian.slack.store import MessageLog, OpenIndex

_WHEN = "2026-10-22T18:00:00+00:00"


def _ports(tmp_path: Path) -> ApprovalPorts:
    return ApprovalPorts(
        gateway=RecordingGateway(MessageLog(tmp_path / "messages")),
        apply_root=tmp_path / "applied",
        postmortem_root=tmp_path / "postmortems",
        channel="C1",
        open_index=OpenIndex(tmp_path / "open.json"),
    )


def _draft(kind: str, source: str) -> PatchDraft:
    return PatchDraft(remediation_kind=kind, module_source=source, regression_test=_CHECK_REPAIR)


def _open(tmp_path: Path, ports: ApprovalPorts, drafts: list[PatchDraft], approved: list[bool]):
    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=FaultType.SCHEMA_DRIFT,
        intensity=0.3,
        seed=3,
        rows=2000,
        name="drift",
    )
    run_id = uuid4()
    config = {"configurable": {"thread_id": str(run_id)}, "recursion_limit": 40}
    saver_cm = SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite"))
    saver = saver_cm.__enter__()
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
        approval=True,
        approval_ports=ports,
        psi_min_rows=200,
    )
    try:
        graph.invoke(MeridianState(run_id=run_id, partitions=[ref]), config)
    except Exception as exc:
        if type(exc).__name__ != "GraphInterrupt":
            saver_cm.__exit__(type(exc), exc, exc.__traceback__)
            raise
    return graph, config, saver_cm, run_id


def _resume(graph, config, saver_cm, payload: dict[str, object]) -> MeridianState:
    try:
        graph.invoke(Command(resume=payload), config)
        values = graph.get_state(config).values
    finally:
        saver_cm.__exit__(None, None, None)
    return MeridianState.model_validate(values)


def _decision(kind: str, comment: str = "") -> dict[str, object]:
    return {
        "decision": kind,
        "approver": "UOWNER",
        "decided_at": _WHEN,
        "comment": comment,
    }


def _case(state: MeridianState):
    return next(iter(state.cases.values()))


def test_approve_after_a_verifier_pass_applies_and_writes_the_postmortem(tmp_path: Path) -> None:
    ports = _ports(tmp_path)
    graph, config, saver, _run_id = _open(
        tmp_path,
        ports,
        [_draft(RemediationKind.DATA_REPAIR.value, _REPAIR)],
        [True],
    )
    gateway = ports.gateway
    assert isinstance(gateway, RecordingGateway)
    assert gateway.posts
    assert button_kinds(gateway.posts[0]["blocks"]) == ["approve", "reject"]
    state = _resume(graph, config, saver, _decision(ApprovalOutcome.APPROVED.value))
    case = _case(state)
    assert case.approval is not None
    assert case.approval.decision is ApprovalOutcome.APPROVED
    assert case.postmortem is not None
    applied = tmp_path / "applied" / str(case.incident.id)
    assert (applied / "manifest.json").is_file()
    assert (applied / "remediation.py").is_file()
    report = (tmp_path / "postmortems" / f"{case.incident.id}.md").read_text(encoding="utf-8")
    assert "UOWNER" in report
    assert "USD:" in report


def test_reject_writes_the_comment_and_does_not_apply(tmp_path: Path) -> None:
    ports = _ports(tmp_path)
    graph, config, saver, _run_id = _open(
        tmp_path,
        ports,
        [_draft(RemediationKind.DATA_REPAIR.value, _REPAIR)],
        [True],
    )
    state = _resume(graph, config, saver, _decision(ApprovalOutcome.REJECTED.value, "too broad"))
    case = _case(state)
    assert case.approval is not None
    assert case.approval.decision is ApprovalOutcome.REJECTED
    assert not (tmp_path / "applied").exists()
    report = (tmp_path / "postmortems" / f"{case.incident.id}.md").read_text(encoding="utf-8")
    assert "too broad" in report


def test_timeout_needs_a_human_and_does_not_apply(tmp_path: Path) -> None:
    ports = _ports(tmp_path)
    graph, config, saver, _run_id = _open(
        tmp_path,
        ports,
        [_draft(RemediationKind.DATA_REPAIR.value, _REPAIR)],
        [True],
    )
    state = _resume(graph, config, saver, _decision(ApprovalOutcome.TIMEOUT.value))
    case = _case(state)
    assert case.needs_human is True
    assert case.approval is not None
    assert case.approval.decision is ApprovalOutcome.TIMEOUT
    assert not (tmp_path / "applied").exists()


def test_escalation_offers_acknowledge_only(tmp_path: Path) -> None:
    ports = _ports(tmp_path)
    graph, config, saver, _run_id = _open(
        tmp_path,
        ports,
        [_draft(RemediationKind.ESCALATE.value, "def repair(df):\n    return df\n")],
        [False],
    )
    gateway = ports.gateway
    assert isinstance(gateway, RecordingGateway)
    assert button_kinds(gateway.posts[0]["blocks"]) == ["acknowledge"]
    state = _resume(graph, config, saver, _decision(ApprovalOutcome.ACKNOWLEDGED.value))
    case = _case(state)
    assert case.needs_human is True
    assert not (tmp_path / "applied").exists()
    assert case.postmortem is not None


def test_restart_resumes_the_checkpoint(tmp_path: Path) -> None:
    first = _ports(tmp_path)
    graph, config, saver, run_id = _open(
        tmp_path,
        first,
        [_draft(RemediationKind.DATA_REPAIR.value, _REPAIR)],
        [True],
    )
    saver.__exit__(None, None, None)
    del graph
    second = ApprovalPorts(
        gateway=RecordingGateway(MessageLog(tmp_path / "messages")),
        apply_root=tmp_path / "applied",
        postmortem_root=tmp_path / "postmortems",
        channel="C1",
        open_index=OpenIndex(tmp_path / "open.json"),
    )
    saver_cm = SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite"))
    restarted = saver_cm.__enter__()
    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=FaultType.SCHEMA_DRIFT,
        intensity=0.3,
        seed=3,
        rows=2000,
        name="drift",
    )
    del ref
    again = build_graph(
        num_cpus=1,
        model_name="fake-triage",
        chat=FakeTriageModel(),
        checkpointer=restarted,
        root_model_name="fake-root-cause",
        root_chat=FakeRootCauseModel(mode="immediate"),
        evidence_root=tmp_path,
        tools=[],
        fix_chat=ScriptedFixModel([]),
        critic_chat=ScriptedCriticModel([]),
        sandbox_timeout_s=30,
        approval=True,
        approval_ports=second,
        psi_min_rows=200,
    )
    state = _resume(again, config, saver_cm, _decision(ApprovalOutcome.APPROVED.value))
    case = _case(state)
    assert case.approval is not None
    assert case.approval.decision is ApprovalOutcome.APPROVED
    assert (tmp_path / "applied" / str(case.incident.id) / "manifest.json").is_file()
    assert run_id
