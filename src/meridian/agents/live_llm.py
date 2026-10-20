"""Live ChatAnthropic client. Constructed only after require_anthropic succeeds."""

from langchain_anthropic import ChatAnthropic

from meridian.config import MeridianConfigError, Settings


def build_live_chat(
    settings: Settings,
    *,
    model: str | None = None,
    model_env: str = "MERIDIAN_TRIAGE_MODEL",
) -> ChatAnthropic:
    """Build a Claude chat model. The caller must already have called ``require_anthropic``.

    Omit ``model`` to use ``MERIDIAN_TRIAGE_MODEL``. Root cause passes its own model name.
    """
    chosen = settings.triage_model if model is None else model
    if chosen is None or chosen.strip() == "":
        raise MeridianConfigError("LLM", [model_env])
    if settings.anthropic_api_key is None:
        raise MeridianConfigError("LLM", ["ANTHROPIC_API_KEY"])
    return ChatAnthropic(
        model=chosen,
        anthropic_api_key=settings.anthropic_api_key,
    )
