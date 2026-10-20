"""Structured-output binding shared by every model node.

Claude 5 and Grok reject forced tool choice when extended thinking is on, and
Claude Sonnet 5.5 rejects it outright. ``json_schema`` uses the provider's native
structured-output API instead of a forced tool call. See ADR-022.
"""

from typing import Any

from pydantic import BaseModel


def bind_structured(chat: Any, schema: type[BaseModel]) -> Any:
    """Bind ``schema`` with native JSON-schema output and the raw response kept."""
    return chat.with_structured_output(schema, include_raw=True, method="json_schema")
