"""Deterministic stand-in for ChatAnthropic. No network."""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ValidationError

_INCIDENT_ID = re.compile(r"Incident-Id: ([0-9a-fA-F-]{36})")
_DOWNSTREAM = re.compile(r"Downstream: (.*)")


class _Bound:
    def __init__(self, model: "FakeTriageModel", schema: type[BaseModel]) -> None:
        self._model = model
        self._schema = schema

    def invoke(self, messages: list[BaseMessage]) -> dict[str, Any]:
        text = self._model.next_text(messages)
        raw = AIMessage(
            content=text,
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )
        try:
            parsed = self._schema.model_validate_json(text)
            error: ValidationError | None = None
        except ValidationError as exc:
            parsed = None
            error = exc
        return {"raw": raw, "parsed": parsed, "parsing_error": error}


class FakeTriageModel:
    """Returns SEV4 for whatever incident id is in the prompt.

    ``failures`` is how many leading calls return text that does not validate.
    """

    def __init__(self, *, failures: int = 0, model_name: str = "fake-triage") -> None:
        self.failures_remaining = failures
        self.model_name = model_name

    def next_text(self, messages: list[BaseMessage]) -> str:
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            return "not-json"
        blob = "\n".join(str(message.content) for message in messages)
        found = _INCIDENT_ID.search(blob)
        incident_id = found.group(1) if found is not None else "not-an-id"
        downstream = _DOWNSTREAM.search(blob)
        consumers: list[str] = []
        if downstream is not None and downstream.group(1).strip() not in {"", "-"}:
            consumers = [part for part in downstream.group(1).split(",") if part]
        body = {
            "incident_id": incident_id,
            "severity": "SEV4",
            "rationale": "Fake model proposes the lowest severity.",
            "affected_consumers": consumers,
            "confidence": 0.25,
        }
        return json.dumps(body)

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **_kwargs: Any,
    ) -> _Bound:
        del include_raw
        return _Bound(self, schema)
