"""JSONL when Postgres is unset or unreachable. Postgres when it answers."""

import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine

from meridian.audit.chain import GENESIS, append_jsonl, digest, read_jsonl, seal, verify_records
from meridian.audit.schema import BY_NAME, create_all

_SQL: "SqlStore | None" = None
_PROBED = False


class JsonlStore:
    """One chain file per stream under a directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(self, stream: str, payload: dict[str, Any]) -> None:
        append_jsonl(self._path(stream), payload)

    def records(self, stream: str) -> list[dict[str, Any]]:
        return read_jsonl(self._path(stream))

    def _path(self, stream: str) -> Path:
        if stream not in BY_NAME:
            raise ValueError(f"unknown audit stream {stream}")
        return self.directory / f"{stream}.jsonl"


class SqlStore:
    """One chain per table. ``id`` order is the chain order."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        create_all(engine)

    def write(self, stream: str, payload: dict[str, Any]) -> None:
        table = BY_NAME[stream]
        with self.engine.begin() as conn:
            prev = _sql_tail(conn, table)
            sealed = seal(prev, payload)
            body = {
                key: value for key, value in sealed.items() if key not in {"prev_hash", "row_hash"}
            }
            conn.execute(
                table.insert().values(
                    prev_hash=sealed["prev_hash"],
                    row_hash=sealed["row_hash"],
                    payload=body,
                )
            )

    def records(self, stream: str) -> list[dict[str, Any]]:
        table = BY_NAME[stream]
        with self.engine.connect() as conn:
            rows = conn.execute(select(table).order_by(table.c.id)).mappings()
            found: list[dict[str, Any]] = []
            for row in rows:
                payload = dict(row["payload"])
                payload["prev_hash"] = row["prev_hash"]
                payload["row_hash"] = row["row_hash"]
                found.append(payload)
            return found


def install(directory: Path, url: str | None = None) -> str:
    """Choose Postgres when ``url`` accepts a connection. Otherwise JSONL.

    Returns ``postgres`` or ``jsonl``. A refused connection does not drop events.
    """
    global _SQL, _PROBED
    _PROBED = True
    if url and _reachable(url):
        _SQL = SqlStore(create_engine(_normalize(url), pool_pre_ping=True))
        return "postgres"
    _SQL = None
    directory.mkdir(parents=True, exist_ok=True)
    return "jsonl"


def install_from_env(directory: Path | None = None) -> str:
    """Use ``DATABASE_URL`` when it is set and the server accepts a connection."""
    url = os.environ.get("DATABASE_URL", "").strip()
    root = directory or Path("data/audit")
    return install(root, url or None)


def sql_store() -> SqlStore | None:
    """The Postgres store, after ``install`` found a server."""
    return _SQL


def note(stream: str, payload: dict[str, Any], *, directory: Path | None = None) -> None:
    """Append one row to the active store. No store means no write."""
    if not _PROBED:
        return
    store = _SQL
    if store is not None:
        store.write(stream, payload)
        return
    root = directory or Path("data/audit")
    JsonlStore(root).write(stream, payload)


def verify_tree(directory: Path) -> list[str]:
    """Verify each JSONL chain. Empty streams are intact."""
    lines: list[str] = []
    root = JsonlStore(directory)
    for stream in BY_NAME:
        path = root._path(stream)
        broken = verify_records(read_jsonl(path))
        if broken is None:
            count = sum(1 for item in read_jsonl(path) if "row_hash" in item)
            lines.append(f"audit chain ok: {count} rows in {path}")
        else:
            lines.append(f"audit chain broken at row {broken} in {path}")
    extra = directory / "slack.jsonl"
    if extra.is_file():
        found = read_jsonl(extra)
        broken_at = verify_records(found)
        if broken_at is None:
            count = sum(1 for item in found if "row_hash" in item)
            lines.append(f"audit chain ok: {count} rows in {extra}")
        else:
            lines.append(f"audit chain broken at row {broken_at} in {extra}")
    return lines


def verify_sql(store: SqlStore) -> list[str]:
    """Verify each Postgres chain."""
    lines: list[str] = []
    for stream in BY_NAME:
        records = store.records(stream)
        broken = verify_records(records)
        if broken is None:
            lines.append(f"audit chain ok: {len(records)} rows in {stream}")
        else:
            lines.append(f"audit chain broken at row {broken} in {stream}")
    return lines


def broken(lines: list[str]) -> bool:
    return any(line.startswith("audit chain broken") for line in lines)


def _sql_tail(conn: Any, table: Any) -> str:
    row = conn.execute(select(table.c.row_hash).order_by(table.c.id.desc()).limit(1)).first()
    if row is None:
        return GENESIS
    return str(row[0])


def reset() -> None:
    """Drop the process store. Tests use this so one case cannot leak into the next."""
    global _SQL, _PROBED
    _SQL = None
    _PROBED = False


def _reachable(url: str) -> bool:
    args: dict[str, object] = {}
    if not url.startswith("sqlite"):
        args["connect_timeout"] = 2
    engine = create_engine(
        _normalize(url),
        pool_pre_ping=True,
        connect_args=args,
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("select 1")
    except Exception:
        return False
    finally:
        engine.dispose()
    return True


def _normalize(url: str) -> str:
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    return url


def recompute(record: dict[str, Any]) -> str:
    """The digest a stored row should have. Used by the tamper test."""
    return digest(str(record["prev_hash"]), record)
