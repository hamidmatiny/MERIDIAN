"""Mention parsing and the status reply. The bot id comes from auth.test."""

import re
from datetime import datetime

from meridian.slack.store import OpenCase

_HELP = "Commands: status, help"
_USAGE = "Usage: status | help"


def strip_bot_mention(text: str, bot_user_id: str) -> str:
    """Remove ``<@BOTID>`` and ``<@BOTID|name>``. The display name is ignored."""
    pattern = re.compile(rf"<@{re.escape(bot_user_id)}(?:\|[^>]+)?>")
    return " ".join(pattern.sub(" ", text).split())


def mention_command(text: str, bot_user_id: str) -> str:
    """Return ``status``, ``help``, or ``usage`` from the mention body."""
    body = strip_bot_mention(text, bot_user_id).strip().lower()
    if body == "status":
        return "status"
    if body == "help":
        return "help"
    return "usage"


def format_status(cases: list[OpenCase], now: datetime) -> str:
    """One line per open case: severity, node, needs_human, and age."""
    if not cases:
        return "No open cases."
    lines: list[str] = []
    for case in cases:
        age = max(int((now - case.detected_at).total_seconds()), 0)
        lines.append(
            f"{case.incident_id} severity={case.severity} node={case.node} "
            f"needs_human={case.needs_human} age_s={age}"
        )
    return "\n".join(lines)


def reply_text(command: str, cases: list[OpenCase], now: datetime) -> str:
    """The thread reply for one mention command."""
    if command == "status":
        return format_status(cases, now)
    if command == "help":
        return _HELP
    return _USAGE
