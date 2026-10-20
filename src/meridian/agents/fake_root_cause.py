"""CLI fake root-cause model. It reads tool JSON only, never a label."""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from meridian.agents.root_cause import RootCauseDraft
from meridian.contracts.hypothesis import Evidence

_USAGE = {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16}


class _Bound:
    def __init__(self, parent: "HeuristicRootCauseModel", *, structured: bool) -> None:
        self._parent = parent
        self._structured = structured

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> object:
        del kwargs
        if self._structured:
            return self._parent._structured_result(messages)
        return self._parent._tool_result(messages)


class HeuristicRootCauseModel:
    """Pick the newest event on the incident dataset from tool output.

    The choice uses ``id``, ``ts``, and ``dataset`` fields returned by tools.
    It does not open a label file and does not read event summaries.
    """

    def __init__(self) -> None:
        self.model_name = "heuristic-root-cause"

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
        dataset = _dataset(messages)
        if not tool_turns:
            return _tool_message("list_events", {"dataset": dataset}, "call-list")
        if len(tool_turns) == 1:
            event_id = _newest_event_id(messages, dataset)
            if event_id is not None:
                return _tool_message("get_event", {"event_id": event_id}, "call-get")
        return AIMessage(content="done", usage_metadata=_USAGE)

    def _structured_result(self, messages: list[BaseMessage]) -> dict[str, object]:
        draft = _draft(messages, _dataset(messages))
        raw = AIMessage(content=draft.model_dump_json(), usage_metadata=_USAGE)
        return {"raw": raw, "parsed": draft, "parsing_error": None}


def _tool_message(name: str, args: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata=_USAGE,
    )


def _draft(messages: list[BaseMessage], dataset: str | None) -> RootCauseDraft:
    event_id = _newest_event_id(messages, dataset)
    if event_id is None:
        return RootCauseDraft(
            root_cause_summary="No event for this dataset was present in the tool output.",
            evidence=[],
            confidence=0.1,
            mechanism="The tools returned no event on the incident dataset.",
        )
    return RootCauseDraft(
        root_cause_summary="The newest event on this dataset is the cited change.",
        evidence=[
            Evidence(
                source="events",
                reference=f"event:{event_id}",
                excerpt="newest event on the dataset",
            )
        ],
        confidence=0.5,
        causal_event_id=event_id,
        mechanism="Selected the event with the latest timestamp on the incident dataset.",
    )


def _dataset(messages: list[BaseMessage]) -> str | None:
    match = re.search(r'"dataset"\s*:\s*"([^"]+)"', _text(messages))
    if match is None:
        return None
    return match.group(1)


def _newest_event_id(messages: list[BaseMessage], dataset: str | None) -> str | None:
    rows: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        rows.extend(_event_rows(payload))
    if dataset is not None:
        rows = [row for row in rows if row["dataset"] == dataset]
    if not rows:
        return None
    rows.sort(key=lambda row: (row["ts"], row["id"]))
    return rows[-1]["id"]


def _event_rows(payload: object) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(payload, dict):
        event_id = payload.get("id")
        timestamp = payload.get("ts")
        dataset = payload.get("dataset")
        if isinstance(event_id, str) and isinstance(timestamp, str) and isinstance(dataset, str):
            found.append({"id": event_id, "ts": timestamp, "dataset": dataset})
        for value in payload.values():
            found.extend(_event_rows(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_event_rows(item))
    return found


def _text(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            parts.append(message.content)
    return "\n".join(parts)
