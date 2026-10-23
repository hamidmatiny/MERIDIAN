"""Files that survive a slack_app restart: decisions, message ts, open cases."""

import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _safe(key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return cleaned or "item"


@dataclass(frozen=True)
class OpenCase:
    """One case waiting on a person."""

    incident_id: str
    severity: str
    node: str
    needs_human: bool
    detected_at: datetime
    posted_at: datetime
    thread_id: str
    patch_id: str

    def to_json(self) -> dict[str, object]:
        return {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "node": self.node,
            "needs_human": self.needs_human,
            "detected_at": self.detected_at.isoformat(),
            "posted_at": self.posted_at.isoformat(),
            "thread_id": self.thread_id,
            "patch_id": self.patch_id,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "OpenCase":
        return cls(
            incident_id=str(raw["incident_id"]),
            severity=str(raw["severity"]),
            node=str(raw["node"]),
            needs_human=bool(raw["needs_human"]),
            detected_at=datetime.fromisoformat(str(raw["detected_at"])),
            posted_at=datetime.fromisoformat(str(raw["posted_at"])),
            thread_id=str(raw["thread_id"]),
            patch_id=str(raw["patch_id"]),
        )


class DecisionLog:
    """One claim per patch id. A second claim is a Slack retry."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def claim(self, patch_id: str) -> bool:
        """Return True only for the first claim of ``patch_id``."""
        path = self.root / _safe(patch_id)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        os.close(fd)
        return True


class MessageLog:
    """Slack ts for a patch, so a resumed process does not post twice."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
        return None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self._path(key).write_text(json.dumps(record), encoding="utf-8")

    def _path(self, key: str) -> Path:
        return self.root / f"{_safe(key)}.json"


class OpenIndex:
    """Open cases the status command reads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def upsert(self, case: OpenCase) -> None:
        with self._lock:
            items = self._load()
            payload = case.to_json()
            current = items.get(case.incident_id)
            if isinstance(current, dict) and isinstance(current.get("posted_at"), str):
                payload["posted_at"] = current["posted_at"]
            items[case.incident_id] = payload
            self._save(items)

    def remove(self, incident_id: str) -> None:
        with self._lock:
            items = self._load()
            items.pop(incident_id, None)
            self._save(items)

    def cases(self) -> list[OpenCase]:
        return [OpenCase.from_json(item) for item in self._load().values()]

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return {}
        found: dict[str, dict[str, Any]] = {}
        for key, value in loaded.items():
            if isinstance(key, str) and isinstance(value, dict):
                found[key] = value
        return found

    def _save(self, items: dict[str, dict[str, Any]]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(items), encoding="utf-8")
        temporary.replace(self.path)
