"""Test-only root-cause model. It may read the causal-event marker in tool JSON.

The CLI fake does not use this module. Modes that know the answer stay under tests/.
"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from meridian.agents.root_cause import RootCauseDraft
from meridian.contracts.hypothesis import Evidence

_USAGE = {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16}
_HALLUCINATED = "00000000-0000-4000-8000-000000000099"


class _Bound:
    def __init__(self, parent: "FakeRootCauseModel", *, structured: bool) -> None:
        self._parent = parent
        self._structured = structured

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> object:
        del kwargs
        if self._structured:
            return self._parent._structured_result(messages)
        return self._parent._tool_result(messages)


class FakeRootCauseModel:
    """Deterministic stand-in for the root-cause chat model.

    ``grounded`` retrieves the causal event and cites it.
    ``hallucinate`` cites an event id no tool returned.
    ``limit`` keeps requesting tools until the loop forces an answer.
    ``immediate`` answers without tools. Graph tests use this mode.
    """

    def __init__(self, mode: str = "grounded") -> None:
        self.mode = mode
        self.model_name = "fake-root-cause"

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> _Bound:
        del tools, kwargs
        return _Bound(self, structured=False)

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> _Bound:
        del schema, include_raw, kwargs
        return _Bound(self, structured=True)

    def _tool_result(self, messages: list[BaseMessage]) -> AIMessage:
        tool_turns = [message for message in messages if isinstance(message, ToolMessage)]
        if self.mode == "raise-model":
            raise RuntimeError("model unavailable")
        if self.mode in {"immediate", "bad-json", "dict-draft", "garbage-id"}:
            return AIMessage(content="done", usage_metadata=_USAGE)
        if self.mode == "limit":
            return _tool_message("list_events", {"dataset": _dataset(messages)}, "limit-call")
        if self.mode == "unknown":
            if not tool_turns:
                return _tool_message("not_a_tool", {}, "call-missing")
            return AIMessage(content="done", usage_metadata=_USAGE)
        if self.mode == "boom":
            if not tool_turns:
                return _tool_message("boom", {}, "call-boom")
            return AIMessage(content="done", usage_metadata=_USAGE)
        if self.mode == "hallucinate" and tool_turns:
            return AIMessage(content="done", usage_metadata=_USAGE)
        if self.mode == "grounded" and len(tool_turns) >= 2:
            return AIMessage(content="done", usage_metadata=_USAGE)
        if not tool_turns:
            return _tool_message("list_events", {"dataset": _dataset(messages)}, "call-list")
        event_id = _causal_id(messages) or _HALLUCINATED
        return _tool_message("get_event", {"event_id": event_id}, "call-get")

    def _structured_result(self, messages: list[BaseMessage]) -> dict[str, object]:
        if self.mode == "bad-json":
            raw = AIMessage(content="not-json", usage_metadata=_USAGE)
            return {"raw": raw, "parsed": None, "parsing_error": ValueError("not-json")}
        if self.mode == "dict-draft":
            draft = _draft("immediate", messages)
            raw = AIMessage(content=draft.model_dump_json(), usage_metadata=_USAGE)
            return {"raw": raw, "parsed": draft.model_dump(), "parsing_error": None}
        if self.mode == "garbage-id":
            draft = RootCauseDraft(
                root_cause_summary="An id that is not a UUID.",
                evidence=[],
                confidence=0.8,
                causal_event_id="not-a-uuid",
                mechanism="Unparseable.",
            )
            raw = AIMessage(content=draft.model_dump_json(), usage_metadata=_USAGE)
            return {"raw": raw, "parsed": draft, "parsing_error": None}
        draft = _draft(self.mode, messages)
        raw = AIMessage(content=draft.model_dump_json(), usage_metadata=_USAGE)
        return {"raw": raw, "parsed": draft, "parsing_error": None}


def _tool_message(name: str, args: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata=_USAGE,
    )


def _draft(mode: str, messages: list[BaseMessage]) -> RootCauseDraft:
    if mode in {"limit", "immediate"}:
        return RootCauseDraft(
            root_cause_summary="No single change was established.",
            evidence=[],
            confidence=0.2,
            mechanism="The tool budget ended before a cause was cited.",
        )
    if mode == "hallucinate":
        return RootCauseDraft(
            root_cause_summary="A change that was not in the log.",
            evidence=[
                Evidence(source="events", reference=f"event:{_HALLUCINATED}", excerpt="invented")
            ],
            confidence=0.8,
            causal_event_id=_HALLUCINATED,
            suspect_columns=["speed_mps"],
            mechanism="Invented event.",
        )
    event_id = _causal_id(messages)
    if event_id is None:
        return RootCauseDraft(
            root_cause_summary="The log did not contain a causal event.",
            evidence=[],
            confidence=0.1,
            mechanism="No causal event was returned.",
        )
    return RootCauseDraft(
        root_cause_summary="The cited change produced the fault.",
        evidence=[Evidence(source="events", reference=f"event:{event_id}", excerpt="causal event")],
        confidence=0.8,
        causal_event_id=event_id,
        suspect_columns=["speed_mps"],
        mechanism="The cited change matches the fault.",
    )


def _dataset(messages: list[BaseMessage]) -> str | None:
    match = re.search(r'"dataset"\s*:\s*"([^"]+)"', _text(messages))
    if match is None:
        return None
    return match.group(1)


def _causal_id(messages: list[BaseMessage]) -> str | None:
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        found = _causal_in(payload)
        if found is not None:
            return found
    return None


def _causal_in(payload: object) -> str | None:
    if isinstance(payload, dict):
        summary = payload.get("summary")
        event_id = payload.get("id")
        if (
            isinstance(summary, str)
            and summary.startswith("Causal event:")
            and isinstance(event_id, str)
        ):
            return event_id
        for value in payload.values():
            found = _causal_in(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _causal_in(item)
            if found is not None:
                return found
    return None


def _text(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            parts.append(message.content)
    return "\n".join(parts)
