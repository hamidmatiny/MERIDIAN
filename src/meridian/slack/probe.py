"""Startup checks. The token itself is never returned."""

from dataclasses import dataclass
from typing import Any

REQUIRED_SCOPES = (
    "chat:write",
    "channels:read",
    "users:read",
    "files:write",
    "app_mentions:read",
)


@dataclass(frozen=True)
class SlackIdentity:
    """What auth.test, the scope header, and the channel lookup returned."""

    bot_user_id: str
    bot_name: str
    team: str
    scopes: frozenset[str]
    channel_id: str
    channel_name: str
    member: bool


def scopes_from(headers: dict[str, str]) -> frozenset[str]:
    """Parse the ``x-oauth-scopes`` response header."""
    raw = headers.get("x-oauth-scopes", "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def missing_scopes(scopes: frozenset[str]) -> list[str]:
    """Required scopes that the bot token does not have, in a stable order."""
    return [name for name in REQUIRED_SCOPES if name not in scopes]


def startup_problem(identity: SlackIdentity, approvers: frozenset[str]) -> str | None:
    """Why slack_app must refuse to start, or None when it may listen."""
    missing = missing_scopes(identity.scopes)
    if missing:
        return "missing Slack scopes: " + ", ".join(missing)
    if not identity.member:
        return (
            f"bot {identity.bot_user_id} is not a member of "
            f"{identity.channel_name} ({identity.channel_id})"
        )
    if not approvers:
        return "MERIDIAN_SLACK_APPROVERS is empty"
    return None


def fetch_identity(bot: Any, channel_id: str) -> SlackIdentity:
    """Call auth.test, conversations.info, and conversations.members.

    ``apps.connections.open`` is a separate call. This function does not open a socket.
    """
    auth = bot.auth_test()
    headers = dict(getattr(auth, "headers", {}) or {})
    info = bot.conversations_info(channel=channel_id)
    listed = bot.conversations_members(channel=channel_id)
    channel = info["channel"]
    members = listed["members"]
    user_id = str(auth["user_id"])
    return SlackIdentity(
        bot_user_id=user_id,
        bot_name=str(auth["user"]),
        team=str(auth["team"]),
        scopes=scopes_from(headers),
        channel_id=channel_id,
        channel_name=str(channel["name"]),
        member=user_id in members,
    )


def connections_open(app_client: Any) -> tuple[bool, str]:
    """Return ``(ok, error)`` from apps.connections.open. The URL is not returned.

    ``apps.connections.open`` takes the app-level token as ``app_token``. A WebClient
    built with that token exposes it as ``token``. A test double with no token is
    called with no arguments.
    """
    token = getattr(app_client, "token", None)
    if isinstance(token, str) and token:
        opened = app_client.apps_connections_open(app_token=token)
    else:
        opened = app_client.apps_connections_open()
    error = opened.get("error")
    return bool(opened.get("ok")), "" if error is None else str(error)
