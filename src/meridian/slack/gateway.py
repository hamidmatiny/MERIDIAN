"""Posting to Slack. The graph calls this; tests use the recording gateway."""

from typing import Any, Protocol

from meridian.slack.store import MessageLog


class SlackGateway(Protocol):
    """The calls the approval node makes. Posting twice for one key is a no-op."""

    def post_once(
        self,
        *,
        key: str,
        channel: str,
        text: str,
        blocks: list[dict[str, Any]],
        thread_ts: str | None = None,
    ) -> str:
        """Return the message ts. A second call with the same key returns the first ts."""

    def upload_once(
        self,
        *,
        key: str,
        channel: str,
        thread_ts: str,
        filename: str,
        content: str,
    ) -> None:
        """Upload the diff once per key."""

    def show_decision(
        self,
        *,
        key: str,
        channel: str,
        text: str,
        blocks: list[dict[str, Any]],
    ) -> None:
        """Replace the original message. The buttons are already gone from ``blocks``."""

    def thread_ts(self, key: str) -> str | None:
        """The ts of the original message, if one was posted."""


class RecordingGateway:
    """In-memory Slack. A shared ``MessageLog`` is what a restart reads."""

    def __init__(self, log: MessageLog) -> None:
        self.log = log
        self.posts: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    def post_once(
        self,
        *,
        key: str,
        channel: str,
        text: str,
        blocks: list[dict[str, Any]],
        thread_ts: str | None = None,
    ) -> str:
        existing = self.log.get(key)
        if existing is not None and existing.get("ts"):
            return str(existing["ts"])
        ts = f"{len(self.posts) + 1}.0"
        self.posts.append(
            {"key": key, "channel": channel, "text": text, "blocks": blocks, "thread_ts": thread_ts}
        )
        record = existing or {}
        record["ts"] = ts
        record["uploaded"] = bool(record.get("uploaded"))
        self.log.put(key, record)
        return ts

    def upload_once(
        self,
        *,
        key: str,
        channel: str,
        thread_ts: str,
        filename: str,
        content: str,
    ) -> None:
        existing = self.log.get(key) or {}
        if existing.get("uploaded"):
            return
        self.uploads.append(
            {
                "key": key,
                "channel": channel,
                "thread_ts": thread_ts,
                "filename": filename,
                "content": content,
            }
        )
        existing["uploaded"] = True
        if "ts" not in existing:
            existing["ts"] = thread_ts
        self.log.put(key, existing)

    def show_decision(
        self,
        *,
        key: str,
        channel: str,
        text: str,
        blocks: list[dict[str, Any]],
    ) -> None:
        self.updates.append({"key": key, "channel": channel, "text": text, "blocks": blocks})

    def thread_ts(self, key: str) -> str | None:
        existing = self.log.get(key)
        if existing is None or not existing.get("ts"):
            return None
        return str(existing["ts"])


class LiveGateway:
    """WebClient calls. The message log stops a resume from posting a second request."""

    def __init__(self, client: Any, log: MessageLog) -> None:
        self.client = client
        self.log = log

    def post_once(
        self,
        *,
        key: str,
        channel: str,
        text: str,
        blocks: list[dict[str, Any]],
        thread_ts: str | None = None,
    ) -> str:
        existing = self.log.get(key)
        if existing is not None and existing.get("ts"):
            return str(existing["ts"])
        kwargs: dict[str, Any] = {"channel": channel, "text": text, "blocks": blocks}
        if thread_ts is not None:
            kwargs["thread_ts"] = thread_ts
        posted = self.client.chat_postMessage(**kwargs)
        ts = str(posted["ts"])
        record = existing or {}
        record["ts"] = ts
        record["uploaded"] = bool(record.get("uploaded"))
        self.log.put(key, record)
        return ts

    def upload_once(
        self,
        *,
        key: str,
        channel: str,
        thread_ts: str,
        filename: str,
        content: str,
    ) -> None:
        existing = self.log.get(key) or {}
        if existing.get("uploaded"):
            return
        self.client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            filename=filename,
            content=content,
        )
        existing["uploaded"] = True
        if "ts" not in existing:
            existing["ts"] = thread_ts
        self.log.put(key, existing)

    def show_decision(
        self,
        *,
        key: str,
        channel: str,
        text: str,
        blocks: list[dict[str, Any]],
    ) -> None:
        existing = self.log.get(key)
        if existing is None or not existing.get("ts"):
            return
        self.client.chat_update(
            channel=channel,
            ts=str(existing["ts"]),
            text=text,
            blocks=blocks,
        )

    def thread_ts(self, key: str) -> str | None:
        existing = self.log.get(key)
        if existing is None or not existing.get("ts"):
            return None
        return str(existing["ts"])
