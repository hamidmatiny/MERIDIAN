"""Model family names used to keep the critic independent of the fixer."""

_FAMILIES = ("qwen", "gemma", "claude", "grok", "llama", "mistral", "phi", "deepseek", "gpt")


def model_family(model: str) -> str:
    """Family of a model id. ``qwen3.5:4b`` and ``qwen3.5:9b`` are both ``qwen``."""
    base = model.split(":", maxsplit=1)[0].strip().lower()
    for name in _FAMILIES:
        if base.startswith(name):
            return name
    return base
