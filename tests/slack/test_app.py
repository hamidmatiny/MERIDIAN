"""slack_app, the live gateway, and the report. No network."""

import json
import signal
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr
from slack_bolt import App
from slack_bolt.request import BoltRequest
from slack_sdk import WebClient
from tests.contracts.builders import (
    make_approval,
    make_hypothesis,
    make_incident,
    make_patch,
    make_verification,
)

from meridian.config import Settings
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.llm.prices import ModelPrice
from meridian.slack.apply import apply_patch
from meridian.slack.blocks import case_view, decided_blocks
from meridian.slack.gate import _moment, run_approval, run_postmortem
from meridian.slack.gateway import LiveGateway, RecordingGateway
from meridian.slack.handlers import NOT_AUTHORIZED, on_action, on_reject_submit
from meridian.slack.probe import connections_open, fetch_identity, scopes_from
from meridian.slack.report import build_postmortem, ledger_usd, render_markdown
from meridian.slack.runtime import ports_for
from meridian.slack.store import DecisionLog, MessageLog, OpenCase, OpenIndex
from meridian.slack.text import format_status, reply_text
from meridian.slack_app import (
    _action,
    _channel,
    _ephemeral,
    _ephemeral_or_modal,
    _install,
    _ray,
    _rejected,
    _secret,
    configure_logging,
    main,
    register,
    serve,
)

_NOW = datetime(2026, 10, 22, 18, 0, tzinfo=UTC)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "slack_bot_token": SecretStr("xoxb-test"),
        "slack_app_token": SecretStr("xapp-test"),
        "slack_signing_secret": SecretStr("sign-test"),
        "slack_approval_channel": "C1",
        "slack_approvers": "UOWNER",
        "checkpoint_path": str(tmp_path / "ckpt.sqlite"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_configure_logging_and_secret_helpers(tmp_path: Path) -> None:
    configure_logging(tmp_path / "logs" / "app.log")
    assert (tmp_path / "logs" / "app.log").exists()
    assert _secret(None) == ""
    assert _secret(SecretStr("xoxb-test")) == "xoxb-test"
    assert _secret("plain") == "plain"
    assert _channel({"channel": {"id": "C1"}}) == "C1"
    assert _channel({"container": {"channel_id": "C2"}}) == "C2"
    assert _channel({}) == ""


def test_ephemeral_notes_and_the_reject_modal() -> None:
    opened: list[dict[str, object]] = []
    notes: list[str] = []

    class _Client:
        def __init__(self) -> None:
            self.views_open = lambda **kwargs: opened.append(kwargs)
            self.chat_postEphemeral = lambda **kwargs: notes.append(str(kwargs["text"]))

    body = {"trigger_id": "trig", "user": {"id": "U1"}, "channel": {"id": "C1"}}
    _ephemeral_or_modal(body, _Client(), {"type": "modal"})
    _ephemeral_or_modal(body, _Client(), "not authorized")
    _ephemeral_or_modal(body, _Client(), "")
    _ephemeral(body, _Client(), "hello")
    assert opened[0]["trigger_id"] == "trig"
    assert notes == ["not authorized", "hello"]


def test_main_refuses_a_missing_token_a_closed_socket_and_a_startup_problem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "meridian.slack_app.get_settings",
        lambda: Settings(_env_file=None),
    )
    assert main(["--log", str(tmp_path / "a.log")]) == 2

    identity = fetch_identity(_Bot(), "C1")
    monkeypatch.setattr("meridian.slack_app.get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr("meridian.slack_app.fetch_identity", lambda *_args: identity)
    monkeypatch.setattr(
        "meridian.slack_app.connections_open",
        lambda _client: (False, "invalid_auth"),
    )
    assert main(["--log", str(tmp_path / "b.log")]) == 2

    monkeypatch.setattr("meridian.slack_app.connections_open", lambda _client: (True, ""))
    monkeypatch.setattr(
        "meridian.slack_app.startup_problem",
        lambda *_args: "MERIDIAN_SLACK_APPROVERS is empty",
    )
    assert main(["--log", str(tmp_path / "c.log")]) == 2

    monkeypatch.setattr("meridian.slack_app.startup_problem", lambda *_args: None)
    monkeypatch.setattr("meridian.slack_app.serve", lambda *_args: 0)
    assert main(["--log", str(tmp_path / "d.log")]) == 0


def test_serve_starts_the_socket_and_stops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    stopped: list[object] = []

    class _Handler:
        def __init__(self, app: App, token: str) -> None:
            del app, token

        def start(self) -> None:
            stopped[0]()  # type: ignore[operator]

        def close(self) -> None:
            stopped.append("closed")

    def _capture(stop: object) -> None:
        stopped.append(stop)

    monkeypatch.setattr("meridian.slack_app.SocketModeHandler", _Handler)
    monkeypatch.setattr("meridian.slack_app._install", _capture)
    monkeypatch.setattr(
        "meridian.slack_app._ray",
        lambda: type("Ray", (), {"is_initialized": lambda self: False})(),
    )
    assert serve(_settings(tmp_path), fetch_identity(_Bot(), "C1")) == 0
    assert "closed" in stopped


def _no_cases() -> list[object]:
    return []


def test_register_dispatches_mention_approve_and_reject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class _SlackResponse(dict[str, object]):
        def __init__(self, payload: dict[str, object]) -> None:
            super().__init__(payload)
            self.headers = {"x-oauth-scopes": "chat:write"}
            self.status_code = 200

    def _api(self: object, api_method: str, **kwargs: object) -> _SlackResponse:
        del self
        calls.append((api_method, kwargs))
        return _SlackResponse(
            {"ok": True, "ts": "9.0", "user_id": "UBOT", "user": "meridian", "team": "T"}
        )

    monkeypatch.setattr("slack_sdk.web.base_client.BaseClient.api_call", _api)
    app = App(client=WebClient(token="xoxb-test"), token_verification_enabled=False)
    resumed: list[dict[str, object]] = []
    register(
        app,
        bot_user_id="UBOT",
        approvers=frozenset({"UOWNER"}),
        decisions=DecisionLog(tmp_path / "decisions"),
        cases=_no_cases,
        resume=lambda thread_id, payload: resumed.append({"thread": thread_id, **payload}),
        timeout_s=86400,
    )
    mention = app.dispatch(
        BoltRequest(
            body={
                "type": "event_callback",
                "event": {
                    "type": "app_mention",
                    "text": "<@UBOT|meridian> help",
                    "ts": "1.2",
                    "channel": "C1",
                    "user": "UOWNER",
                },
            },
            mode="socket_mode",
        )
    )
    assert mention.status == 200
    assert any("Commands: status, help" in str(payload) for _method, payload in calls)
    approve = json.dumps(
        {"thread_id": "thread-1", "patch_id": "patch-1", "incident_id": "i", "kind": "approve"}
    )
    action = app.dispatch(
        BoltRequest(
            body={
                "type": "block_actions",
                "user": {"id": "UOTHER"},
                "channel": {"id": "C1"},
                "message": {"ts": "1.0"},
                "actions": [{"action_id": "meridian_approve", "value": approve}],
            },
            mode="socket_mode",
        )
    )
    assert action.status == 200
    assert any("not authorized" in str(payload) for _method, payload in calls)
    assert resumed == []
    allowed = app.dispatch(
        BoltRequest(
            body={
                "type": "block_actions",
                "user": {"id": "UOWNER"},
                "trigger_id": "trig",
                "channel": {"id": "C1"},
                "message": {"ts": "1.0"},
                "actions": [{"action_id": "meridian_reject", "value": approve}],
            },
            mode="socket_mode",
        )
    )
    assert allowed.status == 200
    assert any("meridian_reject_submit" in str(payload) for _method, payload in calls)
    blank = app.dispatch(
        BoltRequest(
            body={
                "type": "view_submission",
                "user": {"id": "UOWNER"},
                "view": {
                    "type": "modal",
                    "callback_id": "meridian_reject_submit",
                    "private_metadata": approve,
                    "state": {"values": {"comment": {"comment": {"value": "  "}}}},
                },
            },
            mode="socket_mode",
        )
    )
    assert blank.status == 200
    assert "comment" in str(blank.body)
    assert resumed == []
    denied = app.dispatch(
        BoltRequest(
            body={
                "type": "view_submission",
                "user": {"id": "UOTHER"},
                "view": {
                    "type": "modal",
                    "callback_id": "meridian_reject_submit",
                    "private_metadata": approve,
                    "state": {"values": {"comment": {"comment": {"value": "no"}}}},
                },
            },
            mode="socket_mode",
        )
    )
    assert denied.status == 200
    assert any("not authorized" in str(payload) for _method, payload in calls)
    accepted = app.dispatch(
        BoltRequest(
            body={
                "type": "view_submission",
                "user": {"id": "UOWNER"},
                "view": {
                    "type": "modal",
                    "callback_id": "meridian_reject_submit",
                    "private_metadata": approve,
                    "state": {"values": {"comment": {"comment": {"value": "too broad"}}}},
                },
            },
            mode="socket_mode",
        )
    )
    assert accepted.status == 200
    assert resumed[0]["comment"] == "too broad"


def test_action_and_reject_wrappers(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    resumed: list[dict[str, object]] = []
    value = json.dumps(
        {"thread_id": "thread-1", "patch_id": "patch-9", "incident_id": "i", "kind": "reject"}
    )
    body = {
        "user": {"id": "UOWNER"},
        "channel": {"id": "C1"},
        "message": {"ts": "3.0"},
        "actions": [{"action_id": "meridian_reject", "value": value}],
    }
    note = _action(
        body,
        approvers=frozenset({"UOWNER"}),
        decisions=log,
        resume=lambda thread, payload: resumed.append(payload),
    )
    assert isinstance(note, dict)
    assert note["callback_id"] == "meridian_reject_submit"
    missing = _action(
        {"user": {}, "actions": []},
        approvers=frozenset({"UOWNER"}),
        decisions=log,
        resume=lambda thread, payload: resumed.append(payload),
    )
    assert missing == NOT_AUTHORIZED
    metadata = json.dumps({"patch_id": "patch-9", "thread_id": "thread-1"})
    required = _rejected(
        {
            "user": {"id": "UOWNER"},
            "view": {
                "private_metadata": metadata,
                "state": {"values": {"comment": {"comment": {"value": ""}}}},
            },
        },
        approvers=frozenset({"UOWNER"}),
        decisions=log,
        resume=lambda thread, payload: resumed.append(payload),
    )
    assert required == "comment required"
    denied = _rejected(
        {"user": {"id": "UOTHER"}, "view": {}},
        approvers=frozenset({"UOWNER"}),
        decisions=log,
        resume=lambda thread, payload: None,
    )
    assert denied == NOT_AUTHORIZED


def test_install_restores_the_previous_handlers() -> None:
    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    seen: list[int] = []
    try:
        _install(lambda: seen.append(1))
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert seen == [1]
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


def test_ray_module_imports() -> None:
    assert hasattr(_ray(), "is_initialized")


class _Auth(dict[str, str]):
    headers: dict[str, str]


class _Bot:
    def auth_test(self) -> _Auth:
        result = _Auth(user_id="UBOT", user="meridian", team="Acme")
        result.headers = {
            "x-oauth-scopes": (
                "chat:write, channels:read, users:read, files:write, app_mentions:read"
            )
        }
        return result

    def conversations_info(self, channel: str) -> dict[str, object]:
        del channel
        return {"channel": {"name": "approvals"}}

    def conversations_members(self, channel: str) -> dict[str, object]:
        del channel
        return {"members": ["UBOT"]}


class _Socket:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def apps_connections_open(self) -> dict[str, object]:
        return self.payload


def test_probe_reads_identity_and_hides_the_socket_url() -> None:
    identity = fetch_identity(_Bot(), "C1")
    assert identity.bot_user_id == "UBOT"
    assert identity.bot_name == "meridian"
    assert identity.team == "Acme"
    assert identity.channel_name == "approvals"
    assert identity.member is True
    assert "chat:write" in identity.scopes
    assert scopes_from({}) == frozenset()

    class _AppToken:
        token = "app-token"

        def apps_connections_open(self, *, app_token: str) -> dict[str, object]:
            assert app_token == "app-token"
            return {"ok": True, "url": "wss://secret"}

    assert connections_open(_AppToken()) == (True, "")
    assert connections_open(_Socket({"ok": True, "url": "wss://secret"})) == (True, "")
    assert connections_open(_Socket({"ok": False, "error": "invalid_auth"})) == (
        False,
        "invalid_auth",
    )


def test_gateway_posts_once_and_the_live_client_updates(tmp_path: Path) -> None:
    recorded = RecordingGateway(MessageLog(tmp_path / "recorded"))
    assert recorded.thread_ts("missing") is None
    first = recorded.post_once(key="patch", channel="C1", text="Approval", blocks=[])
    assert recorded.post_once(key="patch", channel="C1", text="Approval", blocks=[]) == first
    recorded.upload_once(
        key="patch",
        channel="C1",
        thread_ts=first,
        filename="a.diff",
        content="diff",
    )
    recorded.upload_once(
        key="patch",
        channel="C1",
        thread_ts=first,
        filename="a.diff",
        content="diff",
    )
    assert len(recorded.uploads) == 1
    assert recorded.thread_ts("patch") == first
    recorded.show_decision(key="patch", channel="C1", text="Decision", blocks=[])
    recorded.upload_once(
        key="bare",
        channel="C1",
        thread_ts="2.0",
        filename="b.diff",
        content="diff",
    )
    assert recorded.thread_ts("bare") == "2.0"

    seen: list[str] = []

    class _Client:
        def __init__(self) -> None:
            self.chat_postMessage = self._post
            self.files_upload_v2 = self._upload
            self.chat_update = self._update

        def _post(self, **kwargs: object) -> dict[str, str]:
            del kwargs
            seen.append("post")
            return {"ts": "4.0"}

        def _upload(self, **kwargs: object) -> None:
            del kwargs
            seen.append("upload")

        def _update(self, **kwargs: object) -> None:
            del kwargs
            seen.append("update")

    live = LiveGateway(_Client(), MessageLog(tmp_path / "live"))
    assert live.thread_ts("patch") is None
    assert live.show_decision(key="patch", channel="C1", text="Decision", blocks=[]) is None
    ts = live.post_once(key="patch", channel="C1", text="Approval", blocks=[], thread_ts="1.0")
    assert live.post_once(key="patch", channel="C1", text="Approval", blocks=[]) == ts
    live.upload_once(key="patch", channel="C1", thread_ts=ts, filename="a.diff", content="diff")
    live.upload_once(key="patch", channel="C1", thread_ts=ts, filename="a.diff", content="diff")
    live.show_decision(key="patch", channel="C1", text="Decision", blocks=[])
    assert seen == ["post", "upload", "update"]
    assert live.thread_ts("patch") == ts


def test_store_ignores_a_bad_file_and_a_blank_key(tmp_path: Path) -> None:
    log = MessageLog(tmp_path / "messages")
    assert log.get("missing") is None
    (log.root / "bad.json").write_text("[]", encoding="utf-8")
    assert log.get("bad") is None
    log.put("a/b", {"ts": "1"})
    assert log.get("a/b") == {"ts": "1"}
    assert DecisionLog(tmp_path / "decisions").claim("") is True
    index = OpenIndex(tmp_path / "open.json")
    index.path.write_text("[]", encoding="utf-8")
    assert index.cases() == []
    case = OpenCase(
        incident_id="inc",
        severity="SEV1",
        node="slack_approval",
        needs_human=True,
        detected_at=_NOW,
        posted_at=_NOW,
        thread_id="thread",
        patch_id="patch",
    )
    index.upsert(case)
    index.remove("missing")
    assert index.cases()[0].incident_id == "inc"
    index.remove("inc")
    assert index.cases() == []


def test_status_text_and_a_case_view() -> None:
    case = OpenCase(
        incident_id="inc",
        severity="SEV2",
        node="slack_approval",
        needs_human=False,
        detected_at=datetime(2026, 10, 22, 17, 0, tzinfo=UTC),
        posted_at=_NOW,
        thread_id="thread",
        patch_id="patch",
    )
    text = format_status([case], _NOW)
    assert "severity=SEV2" in text
    assert "age_s=3600" in text
    assert reply_text("help", [], _NOW) == "Commands: status, help"
    assert reply_text("other", [], _NOW) == "Usage: status | help"
    incident = make_incident()
    view = case_view(IncidentCase(incident=incident))
    assert view.severity == "unset"
    assert view.remediation_kind == "none"
    assert view.checks == "none"
    rendered = json.dumps(
        decided_blocks(
            view,
            make_approval(patch_id=uuid4()).model_copy(update={"comment": "a < b & c"}),
        )
    )
    assert "&amp;" in rendered
    assert "&lt;" in rendered


def test_apply_and_postmortem_cover_the_empty_and_priced_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident = make_incident()
    bare = IncidentCase(incident=incident)
    with pytest.raises(ValueError, match="apply requires"):
        apply_patch(bare, tmp_path)
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id).model_copy(
        update={
            "remediation_kind": RemediationKind.DATA_REPAIR,
            "regression_test": "def check():\n    return 1",
        }
    )
    approval = make_approval(patch_id=patch.id).model_copy(update={"comment": "ship it"})
    verification = make_verification(patch_id=patch.id, passed=True, failures_remaining=0)
    ready = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        approval=approval,
        verification=verification,
    )
    folder = apply_patch(ready, tmp_path)
    assert (folder / "manifest.json").exists()
    assert (folder / "remediation.py").read_text(encoding="utf-8").endswith("\n")
    assert (folder / "test_regression.py").read_text(encoding="utf-8").endswith("\n")
    empty = build_postmortem(bare, [])
    assert empty.root_cause == "No hypothesis was recorded."
    assert empty.fix_summary == "No patch was produced."
    assert empty.time_to_resolution_s == 0
    priced = LLMCall(
        node="fix",
        model="claude",
        provider="anthropic",
        prompt_version="v1",
        input_tokens=1_000_000,
        output_tokens=0,
        latency_ms=1,
        attempt=1,
        succeeded=True,
    )
    skipped = priced.model_copy(update={"provider": "ollama", "model": "qwen"})
    monkeypatch.setattr(
        "meridian.slack.report.load_prices",
        lambda: {"claude": ModelPrice(input_per_million=3, output_per_million=15)},
    )
    assert ledger_usd([priced, skipped, priced.model_copy(update={"model": "missing"})]) == 3.0
    written = build_postmortem(ready, [priced])
    markdown = render_markdown(ready, written, [priced])
    assert "ship it" in markdown
    assert "USD: 3.00" in markdown
    assert "human:" in markdown


def test_unknown_action_and_a_bad_button_value(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path)
    assert (
        on_action(
            user_id="UOWNER",
            action_id="meridian_other",
            value=json.dumps({"patch_id": "p", "thread_id": "t"}),
            approvers=frozenset({"UOWNER"}),
            log=log,
            resume=lambda thread, payload: None,
            now=_NOW,
            channel="C1",
            message_ts="1",
        )
        == NOT_AUTHORIZED
    )
    with pytest.raises(ValueError, match="button value"):
        on_action(
            user_id="UOWNER",
            action_id="meridian_approve",
            value="[]",
            approvers=frozenset({"UOWNER"}),
            log=log,
            resume=lambda thread, payload: None,
            now=datetime(2026, 10, 22, 18, 0),
            channel="C1",
            message_ts="1",
        )
    assert (
        on_reject_submit(
            user_id="UOWNER",
            comment="no",
            metadata="[]",
            approvers=frozenset({"UOWNER"}),
            log=log,
            resume=lambda thread, payload: None,
            now=_NOW,
        )
        == "comment required"
    )


def test_approval_refuses_a_branch_with_two_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ports = ports_for(_settings(tmp_path), client=object())
    first = IncidentCase(incident=make_incident())
    second = IncidentCase(incident=make_incident())
    state = MeridianState(
        run_id=uuid4(),
        cases={str(first.incident.id): first, str(second.incident.id): second},
    )
    with pytest.raises(ValueError, match="one incident"):
        run_approval(state, ports)
    with pytest.raises(ValueError, match="one incident"):
        run_postmortem(state, ports)
    aware = datetime(2026, 10, 22, 18, 0, tzinfo=UTC)
    assert _moment(aware) == aware
    assert _moment(datetime(2026, 10, 22, 18, 0)).tzinfo is UTC


def test_ports_use_the_injected_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ports = ports_for(_settings(tmp_path), client=object())
    assert ports.channel == "C1"
    assert ports.apply_root == Path("remediations/applied")
