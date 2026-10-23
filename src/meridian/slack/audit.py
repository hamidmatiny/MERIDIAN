"""Append-only Slack audit log. Tokens are never written."""

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EVENTS = frozenset(
    {
        "mention_received",
        "reply_sent",
        "message_posted",
        "file_uploaded",
        "action_received",
        "decision_recorded",
        "duplicate_rejected",
        "unauthorized_rejected",
        "message_updated",
    }
)
_SECRET = re.compile(
    r"(?:xox[baprs]-[A-Za-z0-9-]+|xapp-\d-[A-Za-z0-9-]+|sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{8,})"
)
_DROPPED = frozenset({"token", "secret", "password", "authorization", "api_key", "app_token"})


class AuditLog:
    """One JSON object per line. Existing lines are never rewritten."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: str, **fields: object) -> None:
        """Append one event. Unknown names and secret field names are refused."""
        if event not in _EVENTS:
            raise ValueError(f"unknown audit event {event}")
        record: dict[str, Any] = {"event": event, "at": datetime.now(UTC).isoformat()}
        for key, value in fields.items():
            lowered = key.lower()
            if lowered in _DROPPED or "token" in lowered or "secret" in lowered:
                continue
            record[key] = _redact(value)
        line = json.dumps(record, sort_keys=True, default=str)
        if _SECRET.search(line):
            raise ValueError("audit line still contains a token")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def lines(self) -> list[dict[str, Any]]:
        """Every record currently on disk, in append order."""
        if not self.path.is_file():
            return []
        found: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                found.append(loaded)
        return found


def _redact(value: object) -> object:
    if isinstance(value, str):
        return _SECRET.sub("[redacted]", value)
    return value
