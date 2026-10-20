"""Cap what a model is shown.

Sample rows are limited to 10. A later model turn keeps the latest tool payload
and replaces earlier ones with the refs and a short summary.
"""

import re

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from meridian.contracts.incident import Incident

MAX_SAMPLE_ROWS = 10
_REF = re.compile(r"(?:event|sample|events|psi|lineage|contract|column|partition):[A-Za-z0-9_.:-]+")


def estimate_tokens(text: str) -> int:
    """Rough input size. Four characters to a token, matching the report's before/after."""
    if text == "":
        return 0
    return (len(text) + 3) // 4


def incident_json(incident: Incident) -> tuple[str, str]:
    """Return the JSON sent to a model, then the uncut JSON."""
    full = incident.model_dump_json()
    rows = list(incident.sample_failure_rows[:MAX_SAMPLE_ROWS])
    trimmed = incident.model_copy(update={"sample_failure_rows": rows})
    return trimmed.model_dump_json(), full


def tagged_human(sent: str, *, before: int, after: int) -> HumanMessage:
    """A human turn that remembers the uncut and trimmed prompt sizes."""
    return HumanMessage(
        content=sent,
        additional_kwargs={"prompt_tokens_before": before, "prompt_tokens_after": after},
    )


def prepare_messages(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int, int]:
    """Shrink older tool payloads and return token estimates before and after."""
    shrunk = _shrink(messages)
    return shrunk, _uncut_count(messages), _sent_count(shrunk)


def summarize_tool(name: str, content: object) -> str:
    """Refs plus the payload size. The body is not repeated."""
    text = content if isinstance(content, str) else str(content)
    refs = _REF.findall(text)
    ref_text = ",".join(refs[:8]) if refs else "-"
    return f"ref={ref_text}; tool={name}; summary={len(text)} chars"


def _shrink(messages: list[BaseMessage]) -> list[BaseMessage]:
    indexes = [index for index, message in enumerate(messages) if isinstance(message, ToolMessage)]
    if len(indexes) < 2:
        return list(messages)
    latest = indexes[-1]
    updated = list(messages)
    for index in indexes:
        if index == latest:
            continue
        message = messages[index]
        if not isinstance(message, ToolMessage):
            continue
        updated[index] = ToolMessage(
            content=summarize_tool(str(message.name or ""), message.content),
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
    return updated


def _uncut_count(messages: list[BaseMessage]) -> int:
    total = 0
    for message in messages:
        kwargs = getattr(message, "additional_kwargs", {}) or {}
        if "prompt_tokens_before" in kwargs:
            total += int(kwargs["prompt_tokens_before"])
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        total += estimate_tokens(content)
    return total


def _sent_count(messages: list[BaseMessage]) -> int:
    total = 0
    for message in messages:
        kwargs = getattr(message, "additional_kwargs", {}) or {}
        if "prompt_tokens_after" in kwargs:
            total += int(kwargs["prompt_tokens_after"])
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        total += estimate_tokens(content)
    return total
