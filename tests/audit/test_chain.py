"""The hash chain fails at the row that was rewritten."""

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from meridian.audit.__main__ import main as audit_main
from meridian.audit.chain import append_jsonl, read_jsonl, verify_records
from meridian.audit.schema import AUDIT_EVENTS
from meridian.audit.store import (
    JsonlStore,
    SqlStore,
    broken,
    install,
    install_from_env,
    note,
    reset,
    sql_store,
    verify_sql,
    verify_tree,
)


def test_jsonl_tamper_fails_at_that_row(tmp_path: Path) -> None:
    path = tmp_path / "audit_events.jsonl"
    append_jsonl(path, {"event": "one"})
    append_jsonl(path, {"event": "two"})
    append_jsonl(path, {"event": "three"})
    rows = read_jsonl(path)
    rows[1]["event"] = "tampered"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert verify_records(read_jsonl(path)) == 2


def test_legacy_lines_are_not_links(tmp_path: Path) -> None:
    path = tmp_path / "audit_events.jsonl"
    path.write_text(json.dumps({"event": "message_updated"}) + "\n", encoding="utf-8")
    append_jsonl(path, {"event": "mention_received"})
    assert verify_records(read_jsonl(path)) is None


def test_sql_tamper_fails_at_that_row(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    store = SqlStore(engine)
    store.write("audit_events", {"event": "one"})
    store.write("audit_events", {"event": "two"})
    with engine.begin() as conn:
        conn.execute(
            AUDIT_EVENTS.update()
            .where(AUDIT_EVENTS.c.id == 2)
            .values(payload={"event": "tampered"})
        )
    lines = verify_sql(store)
    assert any(line.startswith("audit chain broken at row 2 in audit_events") for line in lines)


def test_unreachable_database_url_uses_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@127.0.0.1:1/meridian")
    try:
        assert install_from_env(tmp_path) == "jsonl"
        note("runs", {"run_id": "r"}, directory=tmp_path)
        assert verify_records(read_jsonl(tmp_path / "runs.jsonl")) is None
    finally:
        reset()


def test_note_before_install_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset()
    monkeypatch.chdir(tmp_path)
    note("runs", {"run_id": "r"})
    assert not (tmp_path / "data").exists()


def test_verify_cli_reports_jsonl_and_a_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("meridian.audit.__main__.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    try:
        (tmp_path / "slack.jsonl").write_text(
            json.dumps({"event": "message_updated"}) + "\n", encoding="utf-8"
        )
        assert audit_main(["verify", "--dir", str(tmp_path)]) == 0
        first = capsys.readouterr().out
        assert "audit store: jsonl" in first
        assert "slack.jsonl" in first
        path = tmp_path / "runs.jsonl"
        append_jsonl(path, {"run_id": "1"})
        rows = read_jsonl(path)
        rows[0]["run_id"] = "changed"
        path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        assert audit_main(["verify", "--dir", str(tmp_path)]) == 1
        assert "audit chain broken at row 1" in capsys.readouterr().out
    finally:
        reset()


def test_sqlite_install_verifies_clean(tmp_path: Path) -> None:
    try:
        assert install(tmp_path, f"sqlite:///{tmp_path / 'audit.db'}") == "postgres"
        note("runs", {"run_id": "1"})
        store = sql_store()
        assert store is not None
        assert broken(verify_sql(store)) is False
        assert any(line.startswith("audit chain ok") for line in verify_tree(tmp_path))
    finally:
        reset()


def test_verify_cli_uses_sqlite_and_reports_a_missing_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("meridian.audit.__main__.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        assert audit_main(["verify", "--dir", str(tmp_path)]) == 0
        assert "audit store: postgres" in capsys.readouterr().out
        monkeypatch.setattr("meridian.audit.__main__.sql_store", lambda: None)
        assert audit_main(["verify", "--dir", str(tmp_path)]) == 1
        assert "audit store missing" in capsys.readouterr().err
    finally:
        reset()


def test_unknown_stream_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown audit stream"):
        JsonlStore(tmp_path).write("nope", {"event": "x"})
