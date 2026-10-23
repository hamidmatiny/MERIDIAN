"""Button, modal, and mention decisions. These functions do not open a socket."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from meridian.contracts.approval import ApprovalOutcome
from meridian.slack.audit import AuditLog
from meridian.slack.blocks import reject_modal
from meridian.slack.store import DecisionLog, OpenCase
from meridian.slack.text import mention_command, reply_text

NOT_AUTHORIZED = "not authorized"
ALREADY_DECIDED = "already decided"
COMMENT_REQUIRED = "comment required"


def on_mention(
    event: dict[str, Any],
    say: Callable[..., None],
    *,
    bot_user_id: str,
    cases: list[OpenCase],
    now: datetime,
    audit: AuditLog | None = None,
) -> None:
    """Reply in the mention's thread. The command is parsed from the bot id."""
    text = str(event.get("text", ""))
    user = str(event.get("user", ""))
    _audit(audit, "mention_received", text=text, user=user)
    command = mention_command(text, bot_user_id)
    reply = reply_text(command, cases, now)
    _audit(audit, "reply_sent", text=reply)
    say(text=reply, thread_ts=event.get("ts"))


def on_action(
    *,
    user_id: str,
    action_id: str,
    value: str,
    approvers: frozenset[str],
    log: DecisionLog,
    resume: Callable[[str, dict[str, object]], None],
    now: datetime,
    channel: str,
    message_ts: str,
    audit: AuditLog | None = None,
) -> str | dict[str, object]:
    """Approve, acknowledge, or open the reject modal.

    A string return is an ephemeral note. A dict is the reject modal.
    An empty string means the graph was resumed.
    """
    patch_id = _patch_id(value)
    _audit(audit, "action_received", action=action_id, user=user_id, patch_id=patch_id)
    if user_id not in approvers:
        _audit(audit, "unauthorized_rejected", action=action_id, user=user_id, patch_id=patch_id)
        return NOT_AUTHORIZED
    payload = _value(value)
    if action_id == "meridian_reject":
        return reject_modal(
            patch_id=payload["patch_id"],
            thread_id=payload["thread_id"],
            channel=channel,
            message_ts=message_ts,
        )
    decision = _kind(action_id)
    if decision is None:
        _audit(audit, "unauthorized_rejected", action=action_id, user=user_id, patch_id=patch_id)
        return NOT_AUTHORIZED
    if not log.claim(payload["patch_id"]):
        _audit(
            audit,
            "duplicate_rejected",
            action=action_id,
            user=user_id,
            patch_id=payload["patch_id"],
        )
        return ALREADY_DECIDED
    resume(payload["thread_id"], _body(payload["patch_id"], decision, user_id, now, ""))
    _audit(audit, "decision_recorded", action=decision, user=user_id, patch_id=payload["patch_id"])
    return ""


def on_reject_submit(
    *,
    user_id: str,
    comment: str,
    metadata: str,
    approvers: frozenset[str],
    log: DecisionLog,
    resume: Callable[[str, dict[str, object]], None],
    now: datetime,
    audit: AuditLog | None = None,
) -> str:
    """Resume with REJECTED when the comment is present and the user is allowed."""
    if user_id not in approvers:
        _audit(audit, "unauthorized_rejected", action="meridian_reject", user=user_id, patch_id="")
        return NOT_AUTHORIZED
    if not comment.strip():
        return COMMENT_REQUIRED
    payload = json.loads(metadata)
    if not isinstance(payload, dict):
        return COMMENT_REQUIRED
    patch_id = str(payload["patch_id"])
    thread_id = str(payload["thread_id"])
    _audit(audit, "action_received", action="meridian_reject", user=user_id, patch_id=patch_id)
    if not log.claim(patch_id):
        _audit(
            audit,
            "duplicate_rejected",
            action="meridian_reject",
            user=user_id,
            patch_id=patch_id,
        )
        return ALREADY_DECIDED
    resume(
        thread_id,
        _body(patch_id, ApprovalOutcome.REJECTED.value, user_id, now, comment.strip()),
    )
    _audit(
        audit,
        "decision_recorded",
        action=ApprovalOutcome.REJECTED.value,
        user=user_id,
        patch_id=patch_id,
    )
    return ""


def expire_cases(
    *,
    cases: list[OpenCase],
    timeout_s: int,
    now: datetime,
    log: DecisionLog,
    resume: Callable[[str, dict[str, object]], None],
) -> int:
    """Resume each request older than the timeout. A claimed patch is skipped."""
    resumed = 0
    for case in cases:
        age = (now - case.posted_at).total_seconds()
        if age < timeout_s or not case.patch_id:
            continue
        if not log.claim(case.patch_id):
            continue
        resume(
            case.thread_id,
            _body(case.patch_id, ApprovalOutcome.TIMEOUT.value, "timeout", now, ""),
        )
        resumed += 1
    return resumed


def _audit(audit: AuditLog | None, event: str, **fields: object) -> None:
    if audit is not None:
        audit.write(event, **fields)


def _patch_id(raw: str) -> str:
    try:
        return _value(raw).get("patch_id", "")
    except (ValueError, KeyError):
        return ""


def _value(raw: str) -> dict[str, str]:
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("button value must be an object")
    return {str(key): str(item) for key, item in loaded.items()}


def _kind(action_id: str) -> str | None:
    if action_id == "meridian_approve":
        return ApprovalOutcome.APPROVED.value
    if action_id == "meridian_acknowledge":
        return ApprovalOutcome.ACKNOWLEDGED.value
    return None


def _body(
    patch_id: str,
    decision: str,
    approver: str,
    now: datetime,
    comment: str,
) -> dict[str, object]:
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return {
        "patch_id": patch_id,
        "decision": decision,
        "approver": approver,
        "decided_at": moment.isoformat(),
        "comment": comment,
    }
