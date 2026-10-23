"""Shared directories for the CLI and slack_app. Both must see the same files."""

from pathlib import Path
from typing import Any

from slack_sdk import WebClient

from meridian.config import Settings
from meridian.slack.audit import AuditLog
from meridian.slack.gate import ApprovalPorts
from meridian.slack.gateway import LiveGateway
from meridian.slack.store import MessageLog, OpenIndex


def ports_for(settings: Settings, *, client: Any | None = None) -> ApprovalPorts:
    """The apply directory, the postmortem directory, and the Slack client."""
    token = ""
    if settings.slack_bot_token is not None:
        token = settings.slack_bot_token.get_secret_value()
    web = client if client is not None else WebClient(token=token)
    data = Path("data")
    channel = settings.slack_approval_channel or ""
    return ApprovalPorts(
        gateway=LiveGateway(web, MessageLog(data / "slack-messages")),
        apply_root=Path("remediations/applied"),
        postmortem_root=Path("docs/postmortems"),
        channel=channel,
        open_index=OpenIndex(data / "slack-open.json"),
        audit=AuditLog(data / "audit" / "slack.jsonl"),
    )
