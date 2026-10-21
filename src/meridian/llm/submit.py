"""Ask a model to submit a Pydantic contract, by tool call or by JSON mode.

``json_schema`` stays available for callers whose capability probe confirmed it.
This module is the submit-tool and JSON-mode path.
"""

import hashlib
import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ValidationError

from meridian.contracts.llm_call import LLMCall
from meridian.contracts.tool_call import ToolCall
from meridian.llm.errors import call_model
from meridian.llm.usage import assemble_calls

_SUBMIT_NOW = "submit now"
_TEXT_RETRY = "Call the submit tool. Do not reply with plain text."
CRITIC_CORRECTIVE = "call submit_verdict with your decision"
_VALIDATION_RETRIES = 2


def output_method_of(chat: Any) -> str:
    """The method recorded by the capability probe. Fakes stay on JSON schema."""
    value = getattr(chat, "output_method", None)
    if value in {"submit_tool", "json_schema", "json_mode"}:
        return str(value)
    return "json_schema"


def make_submit_tool(name: str, schema: type[BaseModel]) -> BaseTool:
    """A tool whose arguments are the contract the node has to submit."""

    def _accept(**kwargs: Any) -> str:
        return schema.model_validate(kwargs, strict=False).model_dump_json()

    return StructuredTool.from_function(
        func=_accept,
        name=name,
        description=f"Submit the {schema.__name__} contract.",
        args_schema=schema,
    )


def run_submit_only(
    chat: Any,
    messages: list[BaseMessage],
    schema: type[BaseModel],
    *,
    tool_name: str,
    node: str,
    model_name: str,
    prompt_version: str,
    text_retry: str | None = None,
) -> tuple[BaseModel | None, list[LLMCall]]:
    """Triage and critic: only the submit tool, with validation and text retries."""
    if output_method_of(chat) == "json_mode":
        return _json_mode(
            chat,
            messages,
            schema,
            node=node,
            model_name=model_name,
            prompt_version=prompt_version,
        )
    tool = make_submit_tool(tool_name, schema)
    pending = list(messages)
    calls: list[LLMCall] = []
    validation_left = _VALIDATION_RETRIES
    text_left = 1
    pending_corrective = False
    while True:
        response, batch = _invoke(
            chat,
            [tool],
            pending,
            node=node,
            model_name=model_name,
            prompt_version=prompt_version,
            attempt=len(calls) + 1,
            structured=True,
            corrective=pending_corrective,
        )
        pending_corrective = False
        calls.extend(batch)
        call = calls[-1]
        parsed, kind, detail, call_id = _read(response, schema, tool_name)
        if parsed is not None:
            calls[-1] = call.model_copy(update={"succeeded": True})
            return parsed, calls
        if kind == "validation" and validation_left:
            validation_left -= 1
            pending = _with_tool_error(pending, response, tool_name, call_id, detail)
            continue
        if kind == "text" and text_left:
            text_left -= 1
            pending = _with_text_retry(pending, response, text_retry or _TEXT_RETRY)
            pending_corrective = text_retry is not None
            continue
        return None, calls


def run_tool_loop(
    chat: Any,
    messages: list[BaseMessage],
    schema: type[BaseModel],
    *,
    tool_name: str,
    mcp_tools: list[BaseTool],
    node: str,
    model_name: str,
    prompt_version: str,
    max_tool_calls: int,
    submit_guard: Callable[[BaseModel, list[ToolCall]], str | None] | None = None,
) -> tuple[BaseModel | None, list[LLMCall], list[ToolCall], list[str]]:
    """Root cause and fix. The model may submit on any turn. The last turn must."""
    submit = make_submit_tool(tool_name, schema)
    by_name = {tool.name: tool for tool in mcp_tools}
    pending = list(messages)
    calls: list[LLMCall] = []
    records: list[ToolCall] = []
    payloads: list[str] = []
    steps = 0
    validation_left = _VALIDATION_RETRIES
    text_left = 1
    forced = False
    while True:
        last = steps >= max_tool_calls
        bound_tools = [submit] if last else [*mcp_tools, submit]
        if last and not forced:
            pending = [*pending, HumanMessage(content=_SUBMIT_NOW)]
            forced = True
        response, batch = _invoke(
            chat,
            bound_tools,
            pending,
            node=node,
            model_name=model_name,
            prompt_version=prompt_version,
            attempt=len(calls) + 1,
            structured=last,
        )
        calls.extend(batch)
        call = calls[-1]
        parsed, kind, detail, call_id = _read(response, schema, tool_name)
        if parsed is not None:
            reason = None if submit_guard is None else submit_guard(parsed, records)
            if reason is None:
                calls[-1] = call.model_copy(update={"succeeded": True, "structured": True})
                return parsed, calls, records, payloads
            if validation_left:
                validation_left -= 1
                calls[-1] = call.model_copy(update={"structured": True})
                pending = _with_tool_error(pending, response, tool_name, call_id, reason)
                continue
            return None, calls, records, payloads
        if kind == "mcp" and not last:
            payload, record = _run_mcp(detail, _args_of(response), by_name, node)
            records.append(record)
            payloads.append(payload)
            pending = _with_tool_result(pending, response, detail, call_id, payload)
            steps += 1
            text_left = 1
            continue
        if kind == "mcp":
            kind = "text"
        if kind == "validation" and validation_left:
            validation_left -= 1
            calls[-1] = call.model_copy(update={"structured": True})
            pending = _with_tool_error(pending, response, tool_name, call_id, detail)
            continue
        if kind == "text" and text_left:
            text_left -= 1
            calls[-1] = call.model_copy(update={"structured": True})
            pending = _with_text_retry(pending, response)
            continue
        return None, calls, records, payloads


def _json_mode(
    chat: Any,
    messages: list[BaseMessage],
    schema: type[BaseModel],
    *,
    node: str,
    model_name: str,
    prompt_version: str,
) -> tuple[BaseModel | None, list[LLMCall]]:
    pending = list(messages)
    calls: list[LLMCall] = []
    left = _VALIDATION_RETRIES
    while True:
        bound = chat.with_structured_output(schema, include_raw=True, method="json_mode")
        started_messages = pending
        response, before, after = call_model(bound, started_messages)
        parsed = _parsed_payload(response, schema)
        error = ""
        if isinstance(response, dict) and response.get("parsing_error") is not None:
            error = str(response.get("parsing_error"))
        batch = assemble_calls(
            node=node,
            model=model_name,
            prompt_version=prompt_version,
            raw=response.get("raw") if isinstance(response, dict) else response,
            latency_ms=0,
            attempt=len(calls) + 1,
            succeeded=parsed is not None,
            bound=bound,
            tokens_before=before,
            tokens_after=after,
            structured=True,
        )
        calls.extend(batch)
        if parsed is not None:
            return parsed, calls
        if left == 0:
            return None, calls
        left -= 1
        pending = [*pending, HumanMessage(content=error or "The JSON did not match the schema.")]


def _invoke(
    chat: Any,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    *,
    node: str,
    model_name: str,
    prompt_version: str,
    attempt: int,
    structured: bool,
    corrective: bool = False,
) -> tuple[object, list[LLMCall]]:
    bound = chat.bind_tools(tools)
    response, before, after = call_model(bound, messages)
    batch = assemble_calls(
        node=node,
        model=model_name,
        prompt_version=prompt_version,
        raw=response,
        latency_ms=0,
        attempt=attempt,
        succeeded=False,
        bound=bound,
        tokens_before=before,
        tokens_after=after,
        structured=structured,
        corrective=corrective,
    )
    return response, batch


def _read(
    response: object,
    schema: type[BaseModel],
    tool_name: str,
) -> tuple[BaseModel | None, str, str, str]:
    requested = _first(response)
    if requested is None:
        return None, "text", _content(response), ""
    name, args, call_id = requested
    if name != tool_name:
        return None, "mcp", name, call_id
    try:
        return schema.model_validate(args, strict=False), "ok", "", call_id
    except ValidationError as exc:
        return None, "validation", str(exc), call_id


def _first(response: object) -> tuple[str, dict[str, object], str] | None:
    calls = getattr(response, "tool_calls", None)
    if not isinstance(calls, list) or not calls:
        return None
    first = calls[0]
    if not isinstance(first, dict):
        return None
    name = first.get("name")
    args = first.get("args", {})
    call_id = first.get("id") or "tool-call"
    if not isinstance(name, str) or name == "" or not isinstance(args, dict):
        return None
    return name, args, str(call_id)


def _content(response: object) -> str:
    content = getattr(response, "content", "")
    return content if isinstance(content, str) else str(content)


def _args_of(response: object) -> dict[str, object]:
    requested = _first(response)
    if requested is None:
        return {}
    return requested[1]


def _with_tool_error(
    messages: list[BaseMessage],
    response: object,
    name: str,
    call_id: str,
    detail: str,
) -> list[BaseMessage]:
    return _with_tool_result(messages, response, name, call_id, detail)


def _with_tool_result(
    messages: list[BaseMessage],
    response: object,
    name: str,
    call_id: str,
    payload: str,
) -> list[BaseMessage]:
    requested = _first(response)
    args = {} if requested is None else requested[1]
    return [
        *messages,
        AIMessage(
            content=_content(response),
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": call_id or "tool-call",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content=payload, tool_call_id=call_id or "tool-call", name=name),
    ]


def _with_text_retry(
    messages: list[BaseMessage],
    response: object,
    note: str = _TEXT_RETRY,
) -> list[BaseMessage]:
    return [
        *messages,
        AIMessage(content=_content(response)),
        HumanMessage(content=note),
    ]


def _run_mcp(
    name: str,
    args: dict[str, object],
    tools: dict[str, BaseTool],
    node: str,
) -> tuple[str, ToolCall]:
    tool = tools.get(name)
    payload = f"unknown tool {name}"
    ok = False
    if tool is not None:
        try:
            from meridian.mcp_servers.client import invoke_tool

            payload = invoke_tool(tool, args)
            ok = True
        except Exception as exc:
            payload = str(exc)
    digest = hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()
    return payload, ToolCall(
        node=node,
        tool=name or "unknown",
        args_hash=digest,
        ref_count=0,
        latency_ms=0,
        ok=ok,
    )


def _parsed_payload(response: object, schema: type[BaseModel]) -> BaseModel | None:
    if not isinstance(response, dict) or response.get("parsing_error") is not None:
        return None
    candidate = response.get("parsed")
    if isinstance(candidate, schema):
        return candidate
    if isinstance(candidate, dict):
        try:
            return schema.model_validate(candidate, strict=False)
        except ValidationError:
            return None
    return None
