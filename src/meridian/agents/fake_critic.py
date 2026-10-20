"""CLI critic. It approves the heuristic patch and does not see a label."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from meridian.agents.critic import VerdictDraft

_USAGE = {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}


class _Bound:
    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> object:
        del messages, kwargs
        draft = VerdictDraft(approved=True, objections=[])
        raw = AIMessage(content=draft.model_dump_json(), usage_metadata=_USAGE)
        return {"raw": raw, "parsed": draft, "parsing_error": None}


class HeuristicCriticModel:
    """Approve the patch. The CLI fake does not read ground truth."""

    def __init__(self) -> None:
        self.model_name = "heuristic-critic"

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> _Bound:
        del schema, include_raw, kwargs
        return _Bound()
