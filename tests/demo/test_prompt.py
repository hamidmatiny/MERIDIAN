"""The demo names itself as a pipeline walkthrough."""

import sys
from pathlib import Path

import pytest

from meridian.audit.store import reset as reset_audit
from meridian.config import Settings
from meridian.contracts.fault_type import FaultType
from meridian.demo import _BANNER, _ask, _names, _run_fault
from meridian.run import _chat
from meridian.tracing import reset as reset_tracing


def test_banner_says_fake_models() -> None:
    assert _BANNER == "fake models: demonstrates the pipeline, not model quality"


def test_non_tty_approves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERIDIAN_DEMO_DECISION", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert _ask() == "approve"


def test_tty_reads_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERIDIAN_DEMO_DECISION", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["later", "reject"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert _ask() == "reject"


def test_names_lists_files(tmp_path: Path) -> None:
    assert _names(tmp_path / "missing") == "(none)"
    assert _names(tmp_path) == "(none)"
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    assert _names(tmp_path) == "a.md"


def test_one_fault_approves_from_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("MERIDIAN_DEMO_DECISION", "approve")
    reset_tracing()
    reset_audit()
    chats = _chat(Settings.model_construct(), live=False)
    _run_fault(tmp_path / "unit", FaultType.UNIT_MISMATCH, seed=3, chats=chats, rows=400)
    applied = list((tmp_path / "unit" / "applied").glob("*"))
    postmortem = list((tmp_path / "unit" / "postmortems").glob("*"))
    assert applied
    assert postmortem


def test_forced_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERIDIAN_DEMO_DECISION", "reject")
    assert _ask() == "reject"
