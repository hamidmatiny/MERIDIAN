"""Versioned prompt files. The version string is recorded on each triage result."""

from pathlib import Path

TRIAGE_PROMPT_NAME = "triage"
TRIAGE_PROMPT_VERSION = "v1"


def load_prompt(name: str, version: str) -> str:
    """Read ``prompts/<name>.<version>.md`` from this package."""
    path = Path(__file__).with_name(f"{name}.{version}.md")
    if not path.is_file():
        raise FileNotFoundError(f"missing prompt {name}.{version}")
    return path.read_text(encoding="utf-8")
