"""Environment-backed settings.

Importing this module never requires API keys. A caller that needs a provider
or Slack asks explicitly with ``require_anthropic``, ``require_xai``, or ``require_slack``.
"""

from typing import Self

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from meridian.llm.family import model_family

REPAIR_MODULES: tuple[str, ...] = ("pandas", "numpy", "math", "datetime")
REGRESSION_MODULES: tuple[str, ...] = (*REPAIR_MODULES, "pandera")


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
    slack_app_token: SecretStr | None = Field(
        default=None,
        description="Slack app-level token used by the approval gate.",
    )
    slack_signing_secret: SecretStr | None = Field(
        default=None,
        description="Slack signing secret used to verify inbound requests.",
    )
    slack_approval_channel: str | None = Field(
        default=None,
        description="Channel ID where patch approvals are requested.",
    )
    slack_approvers: str = Field(
        default="",
        validation_alias=AliasChoices("MERIDIAN_SLACK_APPROVERS"),
        description="Comma-separated Slack user ids allowed to approve, reject, or acknowledge.",
    )
    approval_timeout_s: int = Field(
        default=86400,
        ge=1,
        validation_alias=AliasChoices("MERIDIAN_APPROVAL_TIMEOUT_S"),
        description="Seconds an approval request waits before it becomes TIMEOUT.",
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
    rootcause_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MERIDIAN_ROOTCAUSE_MODEL"),
        description="Model name for root cause. Required only for a live run.",
    )
    fix_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MERIDIAN_FIX_MODEL"),
        description="Model name for the fix agent. Required only for a live run.",
    )
    critic_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MERIDIAN_CRITIC_MODEL"),
        description="Model name for the critic. Required only for a live run.",
    )
    triage_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("MERIDIAN_TRIAGE_PROVIDER"),
        description="Provider for triage: ollama, omniroute, anthropic, xai, or fake.",
    )
    rootcause_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("MERIDIAN_ROOTCAUSE_PROVIDER"),
        description="Provider for root cause.",
    )
    fix_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("MERIDIAN_FIX_PROVIDER"),
        description="Provider for the fix agent.",
    )
    critic_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("MERIDIAN_CRITIC_PROVIDER"),
        description="Provider for the critic. Must be a different model family from the fixer.",
    )
    triage_fallback: str = Field(
        default="ollama,omniroute",
        validation_alias=AliasChoices("MERIDIAN_TRIAGE_FALLBACK"),
        description="Fallback providers for triage. Paid providers are not in the default.",
    )
    rootcause_fallback: str = Field(
        default="ollama,omniroute",
        validation_alias=AliasChoices("MERIDIAN_ROOTCAUSE_FALLBACK"),
        description="Fallback providers for root cause.",
    )
    fix_fallback: str = Field(
        default="ollama,omniroute",
        validation_alias=AliasChoices("MERIDIAN_FIX_FALLBACK"),
        description="Fallback providers for the fix agent.",
    )
    critic_fallback: str = Field(
        default="ollama,omniroute",
        validation_alias=AliasChoices("MERIDIAN_CRITIC_FALLBACK"),
        description="Fallback providers for the critic.",
    )
    llm_seed: int = Field(
        default=42,
        validation_alias=AliasChoices("MERIDIAN_LLM_SEED"),
        description="Seed for every local model call. Temperature is always 0.",
    )
    triage_omniroute_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MERIDIAN_TRIAGE_OMNIROUTE_MODEL"),
        description="Explicit OmniRoute model id for the triage fallback hop.",
    )
    rootcause_omniroute_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MERIDIAN_ROOTCAUSE_OMNIROUTE_MODEL"),
        description="Explicit OmniRoute model id for the root-cause fallback hop.",
    )
    fix_omniroute_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MERIDIAN_FIX_OMNIROUTE_MODEL"),
        description="Explicit OmniRoute model id for the fix fallback hop.",
    )
    critic_omniroute_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MERIDIAN_CRITIC_OMNIROUTE_MODEL"),
        description="Explicit OmniRoute model id for the critic fallback hop.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL"),
        description="Base URL for the local Ollama server.",
    )
    omniroute_base_url: str = Field(
        default="http://localhost:20128/v1",
        validation_alias=AliasChoices("OMNIROUTE_BASE_URL"),
        description="OpenAI-compatible base URL for OmniRoute.",
    )
    omniroute_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OMNIROUTE_API_KEY"),
        description="Bearer token sent to OmniRoute. Never logged.",
    )
    llm_timeout_s: float = Field(
        default=180,
        gt=0,
        validation_alias=AliasChoices("MERIDIAN_LLM_TIMEOUT_S"),
        description="Seconds one model call may run before it is a transient timeout.",
    )
    llm_max_output_tokens: int = Field(
        default=2048,
        ge=1,
        validation_alias=AliasChoices("MERIDIAN_LLM_MAX_OUTPUT_TOKENS"),
        description="Maximum output tokens for one call. Ollama receives this as num_predict.",
    )
    llm_num_ctx: int = Field(
        default=8192,
        ge=1,
        validation_alias=AliasChoices("MERIDIAN_LLM_NUM_CTX"),
        description="Context window for local Ollama models.",
    )
    run_wall_budget_s: float = Field(
        default=900,
        gt=0,
        validation_alias=AliasChoices("MERIDIAN_RUN_WALL_BUDGET_S"),
        description="Seconds one graph run may take before it aborts as WALL_BUDGET.",
    )
    max_model_gb: float = Field(
        default=26,
        gt=0,
        validation_alias=AliasChoices("MERIDIAN_MAX_MODEL_GB"),
        description="Largest combined size of the configured local models, in gigabytes.",
    )
    allow_paid: bool = Field(
        default=False,
        validation_alias=AliasChoices("MERIDIAN_ALLOW_PAID"),
        description="When false, anthropic and xai are refused.",
    )
    max_usd_per_run: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("MERIDIAN_MAX_USD_PER_RUN"),
        description="USD cap for one run. Required before a paid provider is called.",
    )
    max_usd_per_day: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("MERIDIAN_MAX_USD_PER_DAY"),
        description="USD cap for the local day. Required before a paid provider is called.",
    )
    spend_path: str = Field(
        default="data/spend.sqlite",
        validation_alias=AliasChoices("MERIDIAN_SPEND_PATH"),
        description="SQLite file that stores paid token spend.",
    )
    sandbox_timeout_s: float = Field(
        default=20,
        gt=0,
        validation_alias=AliasChoices("MERIDIAN_SANDBOX_TIMEOUT_S"),
        description="Seconds a sandboxed remediation may run before it is cancelled.",
    )
    sandbox_memory_mb: int = Field(
        default=512,
        ge=1,
        validation_alias=AliasChoices("MERIDIAN_SANDBOX_MEMORY_MB"),
        description="Ray memory scheduling hint for one remediation task, in megabytes.",
    )
    max_row_loss: float = Field(
        default=0.01,
        ge=0,
        le=1,
        validation_alias=AliasChoices("MERIDIAN_MAX_ROW_LOSS"),
        description="Largest share of rows a data repair may drop.",
    )
    max_quarantine: float = Field(
        default=0.25,
        ge=0,
        le=1,
        validation_alias=AliasChoices("MERIDIAN_MAX_QUARANTINE"),
        description="Largest share of rows a quarantine may remove.",
    )
    max_outside_change: float = Field(
        default=0.0,
        ge=0,
        le=1,
        validation_alias=AliasChoices("MERIDIAN_MAX_OUTSIDE_CHANGE"),
        description="Largest share of clean sample rows a data repair may change.",
    )
    psi_min_rows: int = Field(
        default=200,
        ge=1,
        validation_alias=AliasChoices("MERIDIAN_PSI_MIN_ROWS"),
        description="Frames smaller than this skip PSI and record skipped_insufficient_rows.",
    )
    rootcause_max_tool_calls: int = Field(
        default=8,
        ge=1,
        validation_alias=AliasChoices("MERIDIAN_ROOTCAUSE_MAX_TOOL_CALLS"),
        description="Maximum tool calls the root-cause loop may make for one case.",
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
    capabilities_path: str = Field(
        default="data/capabilities.json",
        validation_alias=AliasChoices("MERIDIAN_CAPABILITIES_PATH"),
        description="Cache of provider capability probes, keyed by provider, model, and version.",
    )

    @field_validator(
        "anthropic_api_key",
        "xai_api_key",
        "slack_bot_token",
        "slack_app_token",
        "slack_signing_secret",
        "slack_approval_channel",
        "database_url",
        "triage_model",
        "rootcause_model",
        "fix_model",
        "critic_model",
        "triage_omniroute_model",
        "rootcause_omniroute_model",
        "fix_omniroute_model",
        "critic_omniroute_model",
        "omniroute_api_key",
        "max_usd_per_run",
        "max_usd_per_day",
        mode="before",
    )
    @classmethod
    def empty_string_is_missing(cls, value: object) -> object:
        """Treat blank strings as unset so a placeholder cannot look configured."""
        return _blank_to_none(value)

    @field_validator(
        "triage_provider",
        "rootcause_provider",
        "fix_provider",
        "critic_provider",
        mode="before",
    )
    @classmethod
    def provider_name(cls, value: object) -> object:
        """Blank provider names fall back to the local default."""
        if isinstance(value, str):
            text = value.strip().lower()
            return "ollama" if text == "" else text
        return value

    @field_validator("allow_paid", mode="before")
    @classmethod
    def paid_flag(cls, value: object) -> object:
        """Accept 0 and 1. A non-empty string must not become true by accident."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off", ""}:
                return False
        return value

    @model_validator(mode="after")
    def critic_family_differs(self) -> Self:
        """The critic must not be the same model family as the fixer. See ADR-003."""
        if self.fix_model and self.critic_model:
            fixer = model_family(self.fix_model)
            critic = model_family(self.critic_model)
            if fixer == critic:
                raise ValueError(
                    f"critic family {critic} matches the fixer {self.fix_model}. "
                    "Use a different family. Qwen fix with Gemma critic is the free pairing."
                )
        allowed = {"ollama", "omniroute", "anthropic", "xai", "fake"}
        for label, provider in (
            ("triage", self.triage_provider),
            ("rootcause", self.rootcause_provider),
            ("fix", self.fix_provider),
            ("critic", self.critic_provider),
        ):
            if provider not in allowed:
                raise ValueError(f"unknown {label} provider {provider}")
        return self

    def require_anthropic(self) -> Self:
        """Return self when the Anthropic key is set; otherwise raise."""
        if not _secret_present(self.anthropic_api_key):
            raise MeridianConfigError("Anthropic", ["ANTHROPIC_API_KEY"])
        return self

    def require_xai(self) -> Self:
        """Return self when the xAI key is set; otherwise raise."""
        if not _secret_present(self.xai_api_key):
            raise MeridianConfigError("xAI", ["XAI_API_KEY"])
        return self

    def approver_ids(self) -> frozenset[str]:
        """Slack user ids allowed to decide. Empty when the setting is blank."""
        return frozenset(part.strip() for part in self.slack_approvers.split(",") if part.strip())

    def require_slack(self) -> Self:
        """Return self when the Slack approval gate is fully configured."""
        missing = [
            name
            for name, present in (
                ("SLACK_BOT_TOKEN", _secret_present(self.slack_bot_token)),
                ("SLACK_APP_TOKEN", _secret_present(self.slack_app_token)),
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
