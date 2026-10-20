"""Live ChatAnthropic client. Constructed only after require_anthropic succeeds."""

from langchain_anthropic import ChatAnthropic

from meridian.config import MeridianConfigError, Settings


def build_live_chat(settings: Settings) -> ChatAnthropic:
    """Build the triage model named by ``MERIDIAN_TRIAGE_MODEL``.

    The caller must already have called ``Settings.require_anthropic``.
    """
    if settings.triage_model is None or settings.triage_model.strip() == "":
        raise MeridianConfigError("LLM", ["MERIDIAN_TRIAGE_MODEL"])
    if settings.anthropic_api_key is None:
        raise MeridianConfigError("LLM", ["ANTHROPIC_API_KEY"])
    return ChatAnthropic(
        model=settings.triage_model,
        anthropic_api_key=settings.anthropic_api_key,
    )
