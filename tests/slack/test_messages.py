"""Block Kit, mentions, allowlist, and the recorded skip. No network."""

import json
from datetime import UTC, datetime

from meridian.slack.blocks import (
    ApprovalView,
    approval_blocks,
    button_kinds,
    escalation_blocks,
    reject_modal,
)
from meridian.slack.handlers import (
    ALREADY_DECIDED,
    COMMENT_REQUIRED,
    NOT_AUTHORIZED,
    expire_cases,
    on_action,
    on_mention,
    on_reject_submit,
)
from meridian.slack.probe import SlackIdentity, missing_scopes, startup_problem
from meridian.slack.process import shutdown_process
from meridian.slack.store import DecisionLog, OpenCase
from meridian.slack.text import mention_command

_VIEW = ApprovalView(
    severity="SEV1",
    dataset="fleet_daily_kpi",
    partition="2024-01-01",
    fault="NULL_EXPLOSION",
    blast_radius="none",
    root_cause="nulls",
    evidence="psi:fleet_daily_kpi",
    remediation_kind="QUARANTINE",
    rows_fixed="10",
    rows_changed_outside="0",
    checks="contracts=pass (0 incident(s) remain)",
    verdicts="round 1 approved by gemma",
    patch_id="patch-1",
    incident_id="incident-1",
)
_THREAD = "thread-1"
_NOW = datetime(2026, 10, 22, 18, 0, tzinfo=UTC)


def test_approval_blocks_match_the_snapshot() -> None:
    assert approval_blocks(_VIEW, _THREAD) == [
        {"type": "header", "text": {"type": "plain_text", "text": "Approval SEV1"}},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "dataset=fleet_daily_kpi partition=2024-01-01 fault=NULL_EXPLOSION blast=none"
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "root_cause=nulls evidence=psi:fleet_daily_kpi",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ("remediation_kind=QUARANTINE rows_fixed=10 rows_changed_outside=0"),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "checks=contracts=pass (0 incident(s) remain)",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "verdicts=round 1 approved by gemma",
            },
        },
        {
            "type": "actions",
            "block_id": "meridian_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "action_id": "meridian_approve",
                    "style": "primary",
                    "value": json.dumps(
                        {
                            "thread_id": _THREAD,
                            "patch_id": "patch-1",
                            "incident_id": "incident-1",
                            "kind": "approve",
                        }
                    ),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "action_id": "meridian_reject",
                    "style": "danger",
                    "value": json.dumps(
                        {
                            "thread_id": _THREAD,
                            "patch_id": "patch-1",
                            "incident_id": "incident-1",
                            "kind": "reject",
                        }
                    ),
                },
            ],
        },
    ]


def test_escalation_blocks_have_acknowledge_only() -> None:
    blocks = escalation_blocks(_VIEW, _THREAD)
    assert blocks[0]["text"] == {"type": "plain_text", "text": "Escalation SEV1"}
    assert button_kinds(blocks) == ["acknowledge"]
    assert "approve" not in button_kinds(blocks)


def test_reject_modal_requires_a_comment_field() -> None:
    modal = reject_modal(
        patch_id="patch-1",
        thread_id=_THREAD,
        channel="C1",
        message_ts="1.0",
    )
    assert modal["callback_id"] == "meridian_reject_submit"
    assert modal["blocks"][0]["block_id"] == "comment"


def test_mention_uses_the_bot_id_not_the_display_name() -> None:
    assert mention_command("<@UBOT> status", "UBOT") == "status"
    assert mention_command("<@UBOT|meridian> help", "UBOT") == "help"
    assert mention_command("meridian status", "UBOT") == "usage"
    said: list[tuple[str, object]] = []

    def say(**kwargs: object) -> None:
        said.append((str(kwargs["text"]), kwargs.get("thread_ts")))

    on_mention(
        {"text": "<@UBOT|meridian> status", "ts": "9.0"},
        say,
        bot_user_id="UBOT",
        cases=[],
        now=_NOW,
    )
    assert said == [("No open cases.", "9.0")]


def test_unauthorized_and_duplicate_clicks_do_not_resume(tmp_path) -> None:
    log = DecisionLog(tmp_path)
    resumed: list[dict[str, object]] = []
    value = json.dumps(
        {
            "thread_id": _THREAD,
            "patch_id": "patch-1",
            "incident_id": "incident-1",
            "kind": "approve",
        }
    )

    def resume(thread_id: str, payload: dict[str, object]) -> None:
        del thread_id
        resumed.append(payload)

    denied = on_action(
        user_id="UOTHER",
        action_id="meridian_approve",
        value=value,
        approvers=frozenset({"UOWNER"}),
        log=log,
        resume=resume,
        now=_NOW,
        channel="C1",
        message_ts="1.0",
    )
    assert denied == NOT_AUTHORIZED
    assert resumed == []
    assert (
        on_action(
            user_id="UOWNER",
            action_id="meridian_approve",
            value=value,
            approvers=frozenset({"UOWNER"}),
            log=log,
            resume=resume,
            now=_NOW,
            channel="C1",
            message_ts="1.0",
        )
        == ""
    )
    again = on_action(
        user_id="UOWNER",
        action_id="meridian_approve",
        value=value,
        approvers=frozenset({"UOWNER"}),
        log=log,
        resume=resume,
        now=_NOW,
        channel="C1",
        message_ts="1.0",
    )
    assert again == ALREADY_DECIDED
    assert len(resumed) == 1


def test_reject_without_a_comment_does_not_resume(tmp_path) -> None:
    log = DecisionLog(tmp_path)
    resumed: list[object] = []
    metadata = json.dumps({"patch_id": "patch-1", "thread_id": _THREAD})
    note = on_reject_submit(
        user_id="UOWNER",
        comment="  ",
        metadata=metadata,
        approvers=frozenset({"UOWNER"}),
        log=log,
        resume=lambda thread, payload: resumed.append(payload),
        now=_NOW,
    )
    assert note == COMMENT_REQUIRED
    assert resumed == []
    accepted = on_reject_submit(
        user_id="UOWNER",
        comment="too broad",
        metadata=metadata,
        approvers=frozenset({"UOWNER"}),
        log=log,
        resume=lambda thread, payload: resumed.append(payload),
        now=_NOW,
    )
    assert accepted == ""
    assert resumed[0]["comment"] == "too broad"  # type: ignore[index]


def test_timeout_claims_once(tmp_path) -> None:
    log = DecisionLog(tmp_path)
    case = OpenCase(
        incident_id="inc",
        severity="SEV2",
        node="slack_approval",
        needs_human=False,
        detected_at=_NOW,
        posted_at=datetime(2026, 10, 21, 18, 0, tzinfo=UTC),
        thread_id=_THREAD,
        patch_id="patch-1",
    )
    resumed: list[dict[str, object]] = []

    def resume(thread_id: str, payload: dict[str, object]) -> None:
        del thread_id
        resumed.append(payload)

    assert (
        expire_cases(
            cases=[case],
            timeout_s=3600,
            now=_NOW,
            log=log,
            resume=resume,
        )
        == 1
    )
    assert resumed[0]["decision"] == "TIMEOUT"
    assert (
        expire_cases(
            cases=[case],
            timeout_s=3600,
            now=_NOW,
            log=log,
            resume=resume,
        )
        == 0
    )


def test_startup_refuses_a_missing_scope_or_an_absent_bot() -> None:
    assert missing_scopes(frozenset({"chat:write"})) == [
        "channels:read",
        "users:read",
        "files:write",
        "app_mentions:read",
    ]
    present = SlackIdentity(
        bot_user_id="UBOT",
        bot_name="meridian",
        team="T1",
        scopes=frozenset(
            {
                "chat:write",
                "channels:read",
                "users:read",
                "files:write",
                "app_mentions:read",
            }
        ),
        channel_id="C1",
        channel_name="approvals",
        member=False,
    )
    assert startup_problem(present, frozenset({"U1"})) == (
        "bot UBOT is not a member of approvals (C1)"
    )
    member = SlackIdentity(
        bot_user_id=present.bot_user_id,
        bot_name=present.bot_name,
        team=present.team,
        scopes=present.scopes,
        channel_id=present.channel_id,
        channel_name=present.channel_name,
        member=True,
    )
    assert startup_problem(member, frozenset()) == "MERIDIAN_SLACK_APPROVERS is empty"
    assert startup_problem(member, frozenset({"U1"})) is None


def test_shutdown_closes_the_socket_sessions_and_ray() -> None:
    closed: list[str] = []

    class _Session:
        def close(self) -> None:
            closed.append("session")

    class _Ray:
        def is_initialized(self) -> bool:
            return True

        def shutdown(self) -> None:
            closed.append("ray")

    shutdown_process(
        close_socket=lambda: closed.append("socket"),
        sessions=[_Session()],
        ray_module=_Ray(),
    )
    assert closed == ["socket", "session", "ray"]
