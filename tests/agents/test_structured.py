"""Structured output uses the provider JSON schema, not a forced tool call."""

from pydantic import BaseModel

from meridian.agents.structured import bind_structured


class _Schema(BaseModel):
    value: str


def test_bind_structured_requests_json_schema() -> None:
    seen: dict[str, object] = {}

    class _Chat:
        def with_structured_output(self, schema: type[BaseModel], **kwargs: object) -> str:
            seen["schema"] = schema
            seen["kwargs"] = kwargs
            return "bound"

    assert bind_structured(_Chat(), _Schema) == "bound"
    assert seen["schema"] is _Schema
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["method"] == "json_schema"
    assert kwargs["include_raw"] is True
