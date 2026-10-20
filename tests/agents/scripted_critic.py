"""Scripted critic. Each call consumes the next approve/reject decision."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from meridian.agents.critic import VerdictDraft

_USAGE = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


class ScriptedCriticModel:
    """Return verdicts in order. A missing decision rejects."""

    def __init__(self, approved: list[bool], *, objection: str = "revise the patch") -> None:
        self.approved = list(approved)
        self.objection = objection
        self.model_name = "scripted-critic"

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> "_Bound":
        del schema, include_raw, kwargs
        return _Bound(self)


class _Bound:
    def __init__(self, parent: ScriptedCriticModel) -> None:
        self._parent = parent

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> object:
        del messages, kwargs
        approved = False if not self._parent.approved else self._parent.approved.pop(0)
        objections: list[dict[str, str]] = []
        if not approved:
            objections.append({"kind": "OTHER", "message": self._parent.objection})
        draft = VerdictDraft(approved=approved, objections=objections)
        raw = AIMessage(content=draft.model_dump_json(), usage_metadata=_USAGE)
        return {"raw": raw, "parsed": draft, "parsing_error": None}
