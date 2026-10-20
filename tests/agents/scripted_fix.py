"""Scripted fix model. Each structured call consumes the next draft."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from meridian.agents.fix import PatchDraft

_USAGE = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}


class ScriptedFixModel:
    """Return drafts in order. An empty queue escalates."""

    def __init__(self, drafts: list[PatchDraft], *, tool_name: str | None = None) -> None:
        self.drafts = list(drafts)
        self.tool_name = tool_name
        self.model_name = "scripted-fix"
        self.calls = 0

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> "_Bound":
        del tools, kwargs
        return _Bound(self, structured=False)

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> "_Bound":
        del schema, include_raw, kwargs
        return _Bound(self, structured=True)

    def next_draft(self) -> PatchDraft:
        self.calls += 1
        if not self.drafts:
            return PatchDraft(
                remediation_kind="ESCALATE",
                module_source="def repair(df):\n    return df\n",
                regression_test="def check(df):\n    return None\n",
            )
        return self.drafts.pop(0)


class _Bound:
    def __init__(self, parent: ScriptedFixModel, *, structured: bool) -> None:
        self._parent = parent
        self._structured = structured
        self._used_tool = False

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> object:
        del messages, kwargs
        if self._structured:
            draft = self._parent.next_draft()
            raw = AIMessage(content=draft.model_dump_json(), usage_metadata=_USAGE)
            return {"raw": raw, "parsed": draft, "parsing_error": None}
        if self._parent.tool_name and not self._used_tool:
            self._used_tool = True
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self._parent.tool_name,
                        "args": {"dataset": "vehicle_telemetry"},
                        "id": "scripted-tool",
                        "type": "tool_call",
                    }
                ],
                usage_metadata=_USAGE,
            )
        return AIMessage(content="done", usage_metadata=_USAGE)
