"""``python -m meridian.spend`` prints the paid ledger."""

import runpy

import pytest


def test_spend_module_prints_zero(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MERIDIAN_SPEND_PATH", str(tmp_path / "spend.sqlite"))
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("meridian.spend", run_name="__main__")
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "today anthropic $0.00" in text
    assert "total xai $0.00" in text
