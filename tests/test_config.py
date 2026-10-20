"""Settings load without secrets and fail only when a caller requires them."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from meridian.config import MeridianConfigError, Settings, _secret_present, get_settings

_KEYS = (
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "SLACK_BOT_TOKEN",
    "SLACK_SIGNING_SECRET",
    "SLACK_APPROVAL_CHANNEL",
    "DATABASE_URL",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_import_and_construction_tolerate_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear(monkeypatch)
    monkeypatch.chdir(tmp_path)
    import meridian.config

    settings = meridian.config.Settings(_env_file=None)
    assert settings.anthropic_api_key is None
    assert settings.xai_api_key is None
    assert settings.slack_bot_token is None
    assert settings.slack_signing_secret is None
    assert settings.slack_approval_channel is None
    assert settings.database_url is None


def test_blank_values_are_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.setenv("SLACK_APPROVAL_CHANNEL", "\n")
    monkeypatch.setenv("DATABASE_URL", " ")
    settings = Settings(
        _env_file=None,
        xai_api_key=SecretStr("   "),
        slack_bot_token=SecretStr("xoxb-real"),
    )
    assert settings.anthropic_api_key is None
    assert settings.xai_api_key is None
    assert settings.slack_approval_channel is None
    assert settings.database_url is None
    assert settings.slack_bot_token is not None
    assert settings.slack_bot_token.get_secret_value() == "xoxb-real"


def test_secret_present_helper() -> None:
    assert _secret_present(None) is False
    assert _secret_present(SecretStr("   ")) is False
    assert _secret_present(SecretStr("key")) is True


def test_unrelated_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("MERIDIAN_NOT_A_SETTING", "1")
    Settings(_env_file=None)


def test_require_anthropic_names_a_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(MeridianConfigError, match=r"ANTHROPIC_API_KEY") as caught:
        Settings(_env_file=None).require_anthropic()
    assert caught.value.purpose == "Anthropic"
    assert caught.value.missing == ("ANTHROPIC_API_KEY",)


def test_require_xai_names_a_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with pytest.raises(MeridianConfigError, match=r"XAI_API_KEY") as caught:
        Settings(_env_file=None).require_xai()
    assert caught.value.purpose == "xAI"
    assert caught.value.missing == ("XAI_API_KEY",)


def test_require_anthropic_returns_settings_when_the_key_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = Settings(_env_file=None).require_anthropic()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert "sk-ant-test" not in repr(settings)


def test_require_xai_returns_settings_when_the_key_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    settings = Settings(_env_file=None).require_xai()
    assert settings.xai_api_key is not None
    assert settings.xai_api_key.get_secret_value() == "xai-test"


def test_require_slack_names_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(MeridianConfigError, match=r"SLACK_BOT_TOKEN") as caught:
        Settings(_env_file=None).require_slack()
    assert caught.value.purpose == "Slack"
    assert caught.value.missing == (
        "SLACK_BOT_TOKEN",
        "SLACK_SIGNING_SECRET",
        "SLACK_APPROVAL_CHANNEL",
    )


def test_require_slack_reports_a_partial_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APPROVAL_CHANNEL", "C0123456789")
    with pytest.raises(MeridianConfigError, match=r"SLACK_SIGNING_SECRET") as caught:
        Settings(_env_file=None).require_slack()
    assert caught.value.missing == ("SLACK_SIGNING_SECRET",)


def test_require_slack_returns_settings_when_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing")
    monkeypatch.setenv("SLACK_APPROVAL_CHANNEL", "C0123456789")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/meridian")
    settings = Settings(_env_file=None).require_slack()
    assert settings.slack_approval_channel == "C0123456789"
    assert settings.database_url == "postgresql://localhost/meridian"
    assert "xoxb-test" not in repr(settings)


def test_settings_are_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.database_url = "postgresql://localhost/meridian"


def test_non_string_values_fail_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url=123)  # type: ignore[arg-type]


def test_get_settings_reads_dotenv_without_overriding_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=from-file\nXAI_API_KEY=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    settings = get_settings()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "from-env"
    assert settings.xai_api_key is not None
    assert settings.xai_api_key.get_secret_value() == "from-file"


def test_rootcause_defaults_to_eight_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.delenv("MERIDIAN_ROOTCAUSE_MODEL", raising=False)
    monkeypatch.delenv("MERIDIAN_ROOTCAUSE_MAX_TOOL_CALLS", raising=False)
    settings = Settings(_env_file=None)
    assert settings.rootcause_model is None
    assert settings.rootcause_max_tool_calls == 8
    blank = Settings(_env_file=None, rootcause_model="  ")
    assert blank.rootcause_model is None
