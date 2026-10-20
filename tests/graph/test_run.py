"""CLI defaults to the fake model and treats intensity 0 as a clean partition."""

import runpy
from pathlib import Path

import pytest
from pydantic import SecretStr

from meridian.config import Settings
from meridian.contracts.state import MeridianState
from meridian.run import _chat, main


def test_fake_schema_drift_run_prints_state(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "--dataset",
            "vehicle_telemetry",
            "--fault",
            "SCHEMA_DRIFT",
            "--intensity",
            "0.3",
            "--seed",
            "42",
            "--rows",
            "80",
            "--fake-llm",
        ]
    )
    assert code == 0
    state = MeridianState.model_validate_json(capsys.readouterr().out)
    assert state.cases
    drift = next(
        case
        for case in state.cases.values()
        if case.triage is not None and case.triage.floor_severity.value == "SEV2"
    )
    assert drift.triage is not None
    assert drift.incident.severity is None
    assert drift.severity is not None
    assert drift.severity.value == "SEV2"
    assert drift.triage.llm_severity is not None
    assert drift.triage.llm_severity.value == "SEV4"


def test_cli_fake_runs_after_the_label_file_is_deleted(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    from meridian.run import _prepare

    def wrapped(root: Path, **kwargs: object) -> object:
        ref = _prepare(root, **kwargs)
        label = root / "injected_fault.json"
        label.write_text(
            '{"causal_event_id":"00000000-0000-4000-8000-000000000099"}',
            encoding="utf-8",
        )
        label.unlink()
        assert not label.exists()
        return ref

    monkeypatch.setattr("meridian.run._prepare", wrapped)
    code = main(
        [
            "--dataset",
            "vehicle_telemetry",
            "--fault",
            "SCHEMA_DRIFT",
            "--intensity",
            "0.3",
            "--seed",
            "42",
            "--rows",
            "40",
            "--fake-llm",
        ]
    )
    assert code == 0
    state = MeridianState.model_validate_json(capsys.readouterr().out)
    assert state.cases
    for case in state.cases.values():
        assert case.hypothesis is not None
        assert case.hypothesis.grounding_violations == 0
        assert case.hypothesis.causal_event_id is not None


def test_zero_intensity_skips_injection_and_triage(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "--dataset",
            "fleet_daily_kpi",
            "--fault",
            "NULL_EXPLOSION",
            "--intensity",
            "0.0",
            "--seed",
            "7",
            "--rows",
            "2000",
        ]
    )
    assert code == 0
    state = MeridianState.model_validate_json(capsys.readouterr().out)
    assert state.cases == {}
    assert state.llm_calls == []


def test_live_without_keys_exits(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setenv("MERIDIAN_TRIAGE_MODEL", "")
    monkeypatch.setenv("MERIDIAN_ROOTCAUSE_MODEL", "")
    monkeypatch.setenv("MERIDIAN_FIX_MODEL", "")
    monkeypatch.setenv("MERIDIAN_CRITIC_MODEL", "")
    code = main(
        [
            "--dataset",
            "fleet_daily_kpi",
            "--fault",
            "NULL_EXPLOSION",
            "--intensity",
            "0.0",
            "--seed",
            "1",
            "--rows",
            "10",
            "--live",
        ]
    )
    assert code == 2
    captured = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in captured
    assert "XAI_API_KEY" not in captured


def test_live_builder_reads_the_configured_model(monkeypatch) -> None:
    created: dict[str, object] = {}

    class _Dummy:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    monkeypatch.setattr("meridian.agents.live_llm.ChatAnthropic", _Dummy)
    settings = Settings(
        _env_file=None,
        anthropic_api_key=SecretStr("test-key"),
        triage_model="configured-model",
    )
    from meridian.agents.live_llm import build_live_chat

    build_live_chat(settings)
    assert created["model"] == "configured-model"
    assert created["anthropic_api_key"].get_secret_value() == "test-key"


def test_cli_rejects_intensity_and_seed_outside_range() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--dataset",
                "fleet_daily_kpi",
                "--fault",
                "NULL_EXPLOSION",
                "--intensity",
                "1.5",
                "--seed",
                "1",
                "--rows",
                "10",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--dataset",
                "fleet_daily_kpi",
                "--fault",
                "NULL_EXPLOSION",
                "--intensity",
                "0.1",
                "--seed",
                "-1",
                "--rows",
                "10",
            ]
        )


def test_live_chat_uses_the_configured_model(monkeypatch) -> None:
    monkeypatch.setattr(Settings, "require_anthropic", lambda self: None)
    monkeypatch.setattr(Settings, "require_xai", lambda self: self)
    monkeypatch.setattr("meridian.run.build_live_chat", lambda settings, **kwargs: "live-client")
    monkeypatch.setattr("meridian.run.build_live_critic", lambda settings: "critic-client")
    settings = Settings(
        _env_file=None,
        anthropic_api_key=SecretStr("test-key"),
        xai_api_key=SecretStr("xai-test"),
        triage_model="configured-model",
        rootcause_model="configured-root",
        fix_model="configured-fix",
        critic_model="configured-critic",
    )
    chats = _chat(settings, live=True)
    assert chats.triage_chat == "live-client"
    assert chats.root_chat == "live-client"
    assert chats.fix_chat == "live-client"
    assert chats.critic_chat == "critic-client"
    assert chats.triage_model == "configured-model"
    assert chats.root_model == "configured-root"
    assert chats.fix_model == "configured-fix"
    assert chats.critic_model == "configured-critic"


def test_live_without_xai_exits(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setenv("MERIDIAN_TRIAGE_MODEL", "triage")
    monkeypatch.setenv("MERIDIAN_ROOTCAUSE_MODEL", "root")
    monkeypatch.setenv("MERIDIAN_FIX_MODEL", "fix")
    monkeypatch.setenv("MERIDIAN_CRITIC_MODEL", "critic")
    code = main(
        [
            "--dataset",
            "fleet_daily_kpi",
            "--fault",
            "NULL_EXPLOSION",
            "--intensity",
            "0.0",
            "--seed",
            "1",
            "--rows",
            "10",
            "--live",
        ]
    )
    assert code == 2
    captured = capsys.readouterr().err
    assert "XAI_API_KEY" in captured
    assert "ANTHROPIC_API_KEY" not in captured


def test_module_entrypoint_prints_help(monkeypatch) -> None:
    import sys

    monkeypatch.setattr("sys.argv", ["meridian.run", "--help"])
    monkeypatch.delitem(sys.modules, "meridian.run", raising=False)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("meridian.run", run_name="__main__")
    assert caught.value.code == 0


def test_live_builder_requires_a_model_name(monkeypatch) -> None:
    from meridian.agents.live_llm import build_live_chat
    from meridian.config import MeridianConfigError

    monkeypatch.delenv("MERIDIAN_TRIAGE_MODEL", raising=False)
    settings = Settings(_env_file=None, anthropic_api_key=SecretStr("test-key"))
    with pytest.raises(MeridianConfigError, match="MERIDIAN_TRIAGE_MODEL"):
        build_live_chat(settings)


def test_live_critic_uses_the_configured_model(monkeypatch) -> None:
    import importlib

    from meridian.agents.live_llm import build_live_critic

    created: dict[str, object] = {}

    class _Chat:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: type("Mod", (), {"ChatXAI": _Chat})(),
    )
    settings = Settings(
        _env_file=None,
        xai_api_key=SecretStr("xai-test"),
        critic_model="configured-critic",
    )
    build_live_critic(settings)
    assert created["model_name"] == "configured-critic"


def test_live_builder_requires_an_api_key(monkeypatch) -> None:
    from meridian.agents.live_llm import build_live_chat
    from meridian.config import MeridianConfigError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None, triage_model="configured-model")
    with pytest.raises(MeridianConfigError, match="ANTHROPIC_API_KEY"):
        build_live_chat(settings)
