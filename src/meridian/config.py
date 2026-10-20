"""Environment-backed settings.

Importing this module never requires API keys. A caller that needs a provider
or Slack asks explicitly with ``require_llm`` or ``require_slack``.
"""

from typing import Self

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MeridianConfigError(RuntimeError):
    """Raised when a caller requires configuration that is not set."""

    def __init__(self, purpose: str, missing: list[str]) -> None:
        self.purpose = purpose
        self.missing = tuple(missing)
        names = ", ".join(missing)
        super().__init__(
            f"Missing {purpose} configuration: {names}. "
            "Set them in the environment or in a .env file."
        )


def _blank_to_none(value: object) -> object:
    if isinstance(value, str) and value.strip() == "":
        return None
    if isinstance(value, SecretStr) and value.get_secret_value().strip() == "":
        return None
    return value


def _secret_present(value: SecretStr | None) -> bool:
    if value is None:
        return False
    return bool(value.get_secret_value().strip())


class Settings(BaseSettings):
    """Settings loaded from the process environment and an optional ``.env`` file.

    Every field is optional so a missing key cannot crash import. Validation still
    runs on the values that are present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        env_ignore_empty=True,
        populate_by_name=True,
    )

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Claude via langchain-anthropic.",
    )
    xai_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Grok via langchain-xai.",
    )
    slack_bot_token: SecretStr | None = Field(
        default=None,
        description="Slack bot token used by the approval gate.",
    )
    slack_signing_secret: SecretStr | None = Field(
        default=None,
        description="Slack signing secret used to verify inbound requests.",
    )
    slack_approval_channel: str | None = Field(
        default=None,
        description="Channel ID where patch approvals are requested.",
    )
    database_url: str | None = Field(
        default=None,
        description="Postgres URL reserved for the audit log. Unused in Phase 0.",
    )
    triage_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MERIDIAN_TRIAGE_MODEL"),
        description="Model name for triage. Required only for a live run.",
    )
    ray_num_cpus: int = Field(
        default=2,
        ge=1,
        validation_alias=AliasChoices("RAY_NUM_CPUS", "MERIDIAN_RAY_NUM_CPUS"),
        description="CPUs Ray may use for Sentinel tasks.",
    )
    checkpoint_path: str = Field(
        default="data/meridian-checkpoints.sqlite",
        validation_alias=AliasChoices("MERIDIAN_CHECKPOINT_PATH"),
        description="SQLite file the LangGraph checkpointer writes.",
    )

    @field_validator(
        "anthropic_api_key",
        "xai_api_key",
        "slack_bot_token",
        "slack_signing_secret",
        "slack_approval_channel",
        "database_url",
        "triage_model",
        mode="before",
    )
    @classmethod
    def empty_string_is_missing(cls, value: object) -> object:
        """Treat blank strings as unset so a placeholder cannot look configured."""
        return _blank_to_none(value)

    def require_llm(self) -> Self:
        """Return self when both model-provider keys are set; otherwise raise."""
        missing = [
            name
            for name, present in (
                ("ANTHROPIC_API_KEY", _secret_present(self.anthropic_api_key)),
                ("XAI_API_KEY", _secret_present(self.xai_api_key)),
            )
            if not present
        ]
        if missing:
            raise MeridianConfigError("LLM", missing)
        return self

    def require_slack(self) -> Self:
        """Return self when the Slack approval gate is fully configured."""
        missing = [
            name
            for name, present in (
                ("SLACK_BOT_TOKEN", _secret_present(self.slack_bot_token)),
                ("SLACK_SIGNING_SECRET", _secret_present(self.slack_signing_secret)),
                ("SLACK_APPROVAL_CHANNEL", self.slack_approval_channel is not None),
            )
            if not present
        ]
        if missing:
            raise MeridianConfigError("Slack", missing)
        return self


def get_settings() -> Settings:
    """Load settings from the environment, after reading a local ``.env`` if present.

    ``load_dotenv`` does not override variables that are already set.
    """
    load_dotenv(override=False)
    return Settings()


__all__ = [
    "MeridianConfigError",
    "Settings",
    "get_settings",
]
