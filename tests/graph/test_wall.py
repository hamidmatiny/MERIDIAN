"""The wall budget covers a provider that never returns, and posts an escalation."""

import logging
import threading
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from tests.agents.support import write_partition
from tests.contracts.builders import make_incident

from meridian.config import MeridianConfigError, Settings
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.postmortem import Postmortem
from meridian.contracts.state import MeridianState
from meridian.graph.build import _to_postmortem, _wall, build_graph
from meridian.llm.budget import Budget, SpendStore
from meridian.llm.errors import AbortGate, RunAborted
from meridian.llm.router import FallbackChat, Hop
from meridian.run import _checkpoint_wall, _post_wall, main
from meridian.slack.blocks import button_kinds
from meridian.slack.gate import ApprovalPorts, post_wall_escalation, run_approval
from meridian.slack.gateway import RecordingGateway
from meridian.slack.store import MessageLog, OpenIndex
from meridian.wall import WallClock, call_bounded


def _budget(tmp_path: Path) -> Budget:
    return Budget(
        prices={},
        max_run=None,
        max_day=None,
        store=SpendStore(tmp_path / "ledger.sqlite"),
        allow_paid=False,
    )


def _ports(tmp_path: Path) -> tuple[ApprovalPorts, RecordingGateway]:
    gateway = RecordingGateway(MessageLog(tmp_path / "messages"))
    ports = ApprovalPorts(
        gateway=gateway,
        apply_root=tmp_path / "applied",
        postmortem_root=tmp_path / "postmortems",
        channel="C1",
        open_index=OpenIndex(tmp_path / "open.json"),
    )
    return ports, gateway


class _Forever:
    """A provider whose invoke never returns."""

    def __init__(self) -> None:
        self.entered = threading.Event()

    def invoke(self, messages: object, **kwargs: object) -> str:
        del messages, kwargs
        self.entered.set()
        threading.Event().wait()
        return "never"

    def with_structured_output(self, schema: object, **kwargs: object) -> "_Forever":
        del schema, kwargs
        return self


def test_the_clock_refuses_a_second_start_and_a_worker_thread() -> None:
    idle = WallClock(5)
    idle.disarm()
    idle.stop()
    spent = WallClock(0)
    with pytest.raises(RunAborted):
        spent.check()
    spent.start()
    spent.stop()
    clock = WallClock(5)
    clock.start()
    clock.start()
    try:
        clock.check()
        clock.arm()
        seen: list[str] = []

        def _off_main() -> None:
            try:
                WallClock(1).start()
            except RuntimeError as exc:
                seen.append(str(exc))

        worker = threading.Thread(target=_off_main)
        worker.start()
        worker.join()
        assert seen == ["the wall watchdog starts on the main thread"]
    finally:
        clock.stop()


def test_a_bounded_call_reraises_the_provider_error(tmp_path: Path) -> None:
    class _Boom:
        def invoke(self, messages: object, **kwargs: object) -> str:
            del messages, kwargs
            raise RuntimeError("provider down")

    chat = FallbackChat(
        [Hop("fake", "boom", _Boom())],
        budget=_budget(tmp_path),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
        call_timeout_s=1,
        clock=WallClock(5),
        node="critic",
    )
    with pytest.raises(RuntimeError, match="provider down"):
        chat.invoke(["hello"])


def test_call_bounded_returns_and_reraises(tmp_path: Path) -> None:
    del tmp_path
    clock = WallClock(5)
    assert call_bounded(lambda: 7, clock) == 7

    def _boom() -> int:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        call_bounded(_boom, clock)


def test_a_provider_that_blocks_forever_hits_the_wall(tmp_path: Path) -> None:
    clock = WallClock(0.3)
    client = _Forever()
    chat = FallbackChat(
        [Hop("fake", "block-forever", client)],
        budget=_budget(tmp_path),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
        call_timeout_s=None,
        clock=clock,
        node="triage",
        heartbeat_s=30,
    )
    started = time.monotonic()
    with pytest.raises(RunAborted) as caught:
        chat.invoke(["hello"])
    assert caught.value.kind == "WALL_BUDGET"
    assert time.monotonic() - started < 2
    assert client.entered.is_set()


def test_the_watchdog_interrupts_a_blocked_main_thread() -> None:
    clock = WallClock(0.3)
    clock.start()
    try:
        with pytest.raises(RunAborted) as caught:
            time.sleep(30)
        assert caught.value.kind == "WALL_BUDGET"
    finally:
        clock.stop()


def test_a_heartbeat_is_logged_while_the_call_is_in_flight(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Slow:
        def invoke(self, messages: object, **kwargs: object) -> str:
            del messages, kwargs
            time.sleep(0.2)
            return "ok"

    clock = WallClock(5)
    chat = FallbackChat(
        [Hop("fake", "slow-model", _Slow())],
        budget=_budget(tmp_path),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
        call_timeout_s=2,
        clock=clock,
        node="fix",
        heartbeat_s=0.05,
    )
    caplog.set_level(logging.INFO, logger="meridian.progress")
    assert chat.invoke(["hello"]) == "ok"
    beats = [record.message for record in caplog.records if record.message.startswith("heartbeat")]
    assert beats
    assert "node=fix" in beats[0]
    assert "model=slow-model" in beats[0]
    assert "elapsed_s=" in beats[0]


def test_wall_abort_posts_an_escalation_and_does_not_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _waited(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("waited")

    monkeypatch.setattr("meridian.slack.gate.interrupt", _waited)
    ports, gateway = _ports(tmp_path)
    run_id = make_incident().id
    empty = MeridianState(run_id=run_id, aborted=True, abort_reason="WALL_BUDGET")
    assert run_approval(empty, ports) == {}
    incident = make_incident()
    case = IncidentCase(incident=incident)
    state = MeridianState(
        run_id=incident.id,
        aborted=True,
        abort_reason="WALL_BUDGET",
        cases={str(incident.id): case},
    )
    update = run_approval(state, ports)
    assert update["cases"][str(incident.id)].needs_human is True
    headers = [
        block["text"]["text"]
        for post in gateway.posts
        for block in post["blocks"]
        if block["type"] == "header"
    ]
    assert any(str(header).startswith("Escalation") for header in headers)
    assert all(button_kinds(post["blocks"]) == [] for post in gateway.posts)
    post_wall_escalation(ports, None, run_id)
    assert any(post["key"] == f"wall:{run_id}" for post in gateway.posts)


def test_a_blocked_triage_checkpoints_and_posts_without_buttons(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=FaultType.UNIT_MISMATCH,
        intensity=0.3,
        seed=3,
        rows=40,
        name="wall",
    )
    ports, gateway = _ports(tmp_path)
    chat = _Forever()
    clock = WallClock(20)
    run_id = make_incident().id
    config = {"configurable": {"thread_id": str(run_id)}, "recursion_limit": 40}
    caplog.set_level(logging.INFO, logger="meridian.progress")
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as saver:
        graph = build_graph(
            num_cpus=1,
            model_name="block-forever",
            chat=chat,
            checkpointer=saver,
            root_model_name="unused",
            root_chat=_Forever(),
            evidence_root=tmp_path,
            tools=[],
            sandbox_timeout_s=30,
            wall_clock=clock,
            approval=True,
            approval_ports=ports,
            psi_min_rows=200,
        )
        finished = graph.invoke(
            MeridianState(run_id=run_id, partitions=[ref]),
            config,
        )
        saved = graph.get_state(config)
    state = MeridianState.model_validate(finished)
    assert state.aborted is True
    assert state.abort_reason == "WALL_BUDGET"
    assert chat.entered.is_set()
    checkpoint = MeridianState.model_validate(saved.values)
    assert checkpoint.abort_reason == "WALL_BUDGET"
    assert gateway.posts
    assert all(button_kinds(post["blocks"]) == [] for post in gateway.posts)
    assert any(post["text"] == "Escalation" for post in gateway.posts)
    assert not ports.apply_root.exists()
    assert "node=triage" in caplog.text
    assert "phase=start" in caplog.text
    assert "phase=end" in caplog.text
    assert "run_id=" in caplog.text
    assert "elapsed_s=" in caplog.text


def test_main_aborts_when_startup_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("meridian.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("MERIDIAN_RUN_WALL_BUDGET_S", "0.4")
    monkeypatch.setenv("MERIDIAN_CHECKPOINT_PATH", str(tmp_path / "ckpt.sqlite"))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing")
    monkeypatch.setenv("SLACK_APPROVAL_CHANNEL", "C0123456789")
    monkeypatch.setenv("MERIDIAN_SLACK_APPROVERS", "U_TEST")

    def _block(_settings: object, *, live: bool, clock: object = None) -> object:
        del _settings, live, clock
        time.sleep(30)
        raise AssertionError("startup returned")

    posted: list[object] = []
    monkeypatch.setattr("meridian.run._chat", _block)
    monkeypatch.setattr(
        "meridian.run._post_wall",
        lambda _settings, run_id: posted.append(run_id),
    )
    code = main(
        [
            "--dataset",
            "vehicle_telemetry",
            "--fault",
            "DUPLICATE_KEYS",
            "--intensity",
            "0.1",
            "--seed",
            "1",
            "--rows",
            "10",
            "--slack",
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.strip() == '{"status": "aborted", "reason": "WALL_BUDGET"}'
    assert "node=" not in captured.out
    assert "aborted: WALL_BUDGET" in captured.err
    assert posted


def _argv(*extra: str) -> list[str]:
    return [
        "--dataset",
        "vehicle_telemetry",
        "--fault",
        "DUPLICATE_KEYS",
        "--intensity",
        "0.1",
        "--seed",
        "1",
        "--rows",
        "10",
        *extra,
    ]


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("meridian.config.load_dotenv", lambda *_args, **_kwargs: False)


def test_settings_and_slack_configuration_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate(monkeypatch, tmp_path)

    def _bad() -> Settings:
        raise MeridianConfigError("LLM", ["MERIDIAN_TRIAGE_MODEL"])

    monkeypatch.setattr("meridian.run.get_settings", _bad)
    assert main(_argv()) == 2
    assert "MERIDIAN_TRIAGE_MODEL" in capsys.readouterr().err

    monkeypatch.setattr("meridian.run.get_settings", lambda: Settings(_env_file=None))
    assert main(_argv("--slack")) == 2
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing")
    monkeypatch.setenv("SLACK_APPROVAL_CHANNEL", "C0123456789")
    monkeypatch.setenv("MERIDIAN_SLACK_APPROVERS", "")
    assert main(_argv("--slack")) == 2
    assert "MERIDIAN_SLACK_APPROVERS is empty" in capsys.readouterr().err


def test_slack_wait_is_the_only_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing")
    monkeypatch.setenv("SLACK_APPROVAL_CHANNEL", "C0123456789")
    monkeypatch.setenv("MERIDIAN_SLACK_APPROVERS", "U_TEST")
    monkeypatch.setattr("meridian.run._invoke", lambda *_args, **_kwargs: None)
    assert main(_argv("--slack")) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status": "waiting_for_slack"}'
    assert "waiting for Slack approval" in captured.err
    assert "node=" not in captured.out


def test_a_billing_abort_is_not_reported_as_the_wall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate(monkeypatch, tmp_path)

    def _bill(*_args: object, **_kwargs: object) -> int:
        raise RunAborted("BILLING", "no credit")

    monkeypatch.setattr("meridian.run._execute", _bill)
    with pytest.raises(RunAborted, match="BILLING"):
        main(_argv())


def test_a_failed_wall_post_still_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("MERIDIAN_RUN_WALL_BUDGET_S", "0.4")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing")
    monkeypatch.setenv("SLACK_APPROVAL_CHANNEL", "C0123456789")
    monkeypatch.setenv("MERIDIAN_SLACK_APPROVERS", "U_TEST")

    def _block(_settings: object, *, live: bool, clock: object = None) -> object:
        del _settings, live, clock
        time.sleep(30)
        raise AssertionError("startup returned")

    def _explode(_settings: object, _run_id: object) -> None:
        del _settings, _run_id
        raise RuntimeError("slack down")

    monkeypatch.setattr("meridian.run._chat", _block)
    monkeypatch.setattr("meridian.run._post_wall", _explode)
    assert main(_argv("--slack")) == 1
    assert "wall escalation was not posted" in capsys.readouterr().err


def test_checkpoint_and_post_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _checkpoint_wall(object(), {})

    class _Locked:
        def update_state(self, thread: object, values: object) -> None:
            del thread, values
            raise RuntimeError("locked")

    caplog.set_level(logging.WARNING, logger="meridian.run")
    _checkpoint_wall(_Locked(), {"configurable": {"thread_id": "t"}})
    assert "wall checkpoint was not updated" in caplog.text

    ports, gateway = _ports(tmp_path)
    monkeypatch.setattr("meridian.slack.runtime.ports_for", lambda _settings: ports)
    run_id = make_incident().id
    _post_wall(Settings(_env_file=None), run_id)
    assert gateway.posts[-1]["key"] == f"wall:{run_id}"

    incident = make_incident()
    done = IncidentCase(
        incident=incident,
        postmortem=Postmortem(
            incident_id=incident.id,
            root_cause="cause",
            fix_summary="none",
            time_to_resolution_s=0,
            cost_usd=0.0,
        ),
    )
    state = MeridianState(run_id=incident.id, cases={str(incident.id): done})
    assert _to_postmortem(state) == "__end__"
    assert _wall(AbortGate(), state, None) == {"aborted": True, "abort_reason": "WALL_BUDGET"}
    bare = IncidentCase(incident=make_incident())
    pending = MeridianState(run_id=bare.incident.id, cases={str(bare.incident.id): bare})
    assert isinstance(_to_postmortem(pending), list)


def test_a_second_alarm_does_not_raise() -> None:
    clock = WallClock(5)
    clock.trip()
    clock._on_alarm(14, None)
