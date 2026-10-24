"""Hash chain shared by the JSONL fallback and Postgres.

``row_hash`` is ``sha256(prev_hash + canonical JSON)``. The first row uses 64 zeroes
as ``prev_hash``. Lines written before the chain have no ``row_hash`` and are skipped.
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
_HASH_KEYS = frozenset({"prev_hash", "row_hash"})
_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def canonical(payload: dict[str, Any]) -> str:
    """Stable JSON. Key order and spacing do not change the digest."""
    body = {key: value for key, value in payload.items() if key not in _HASH_KEYS}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


def digest(prev_hash: str, payload: dict[str, Any]) -> str:
    """sha256 of the previous digest and the canonical payload."""
    raw = (prev_hash + canonical(payload)).encode()
    return hashlib.sha256(raw).hexdigest()


def seal(prev_hash: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload plus its chain fields."""
    body = {key: value for key, value in payload.items() if key not in _HASH_KEYS}
    return {**body, "prev_hash": prev_hash, "row_hash": digest(prev_hash, body)}


def append_jsonl(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Append one sealed object. Existing lines are not rewritten."""
    with _lock(path):
        prev = _tail_hash(path)
        sealed = seal(prev, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(sealed, sort_keys=True, default=str)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return sealed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every JSON object in the file, in append order."""
    if not path.is_file():
        return []
    found: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            found.append(loaded)
    return found


def verify_records(records: list[dict[str, Any]]) -> int | None:
    """1-based index of the first broken hashed row. None means the chain holds.

    The index counts hashed rows only. Unhashed legacy lines are not links.
    """
    expected = GENESIS
    seen = 0
    for record in records:
        if "row_hash" not in record or "prev_hash" not in record:
            continue
        seen += 1
        prev = str(record["prev_hash"])
        if prev != expected or str(record["row_hash"]) != digest(prev, record):
            return seen
        expected = str(record["row_hash"])
    return None


def _tail_hash(path: Path) -> str:
    prev = GENESIS
    for record in read_jsonl(path):
        if "row_hash" in record:
            prev = str(record["row_hash"])
    return prev


def _lock(path: Path) -> threading.Lock:
    key = str(path)
    with _GUARD:
        found = _LOCKS.get(key)
        if found is None:
            found = threading.Lock()
            _LOCKS[key] = found
        return found
