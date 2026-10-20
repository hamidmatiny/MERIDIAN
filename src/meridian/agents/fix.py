"""Fix agent. Claude proposes a remediation, with the same tool budget as root cause."""

import hashlib
import json
import time
from string import Template
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, ValidationError

from meridian.agents.patch_text import unified_diff
from meridian.agents.root_cause import ToolChat, collect_refs
from meridian.agents.structured import bind_structured
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.patch import Patch
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.contracts.tool_call import ToolCall
from meridian.graph.limits import (
    begin_fix,
    escalate_source,
    finish_fix,
    kind_or_escalate,
    patch_id_for,
)
from meridian.mcp_servers.client import invoke_tool
from meridian.prompts import FIX_PROMPT_NAME, FIX_PROMPT_VERSION, load_prompt

NODE = "fix"


class PatchDraft(BaseModel):
    """Fields the fix model fills. The node assigns ids and the diff."""

    model_config = ConfigDict(extra="ignore")

    remediation_kind: str
    module_source: str
    regression_test: str


def render_fix_prompt(
    state: MeridianState,
    *,
    feedback: str,
    max_row_loss: float,
    max_quarantine: float,
) -> str:
    """Fill the versioned fix prompt for the single case in this branch."""
    _key, case = next(iter(state.cases.items()))
    hypothesis = "{}" if case.hypothesis is None else case.hypothesis.model_dump_json()
    triage = "{}" if case.triage is None else case.triage.model_dump_json()
    template = Template(load_prompt(FIX_PROMPT_NAME, FIX_PROMPT_VERSION))
    return template.substitute(
        incident_json=case.incident.model_dump_json(),
        triage_json=triage,
        hypothesis_json=hypothesis,
        feedback=feedback or "none",
        max_row_loss=f"{max_row_loss:.0%}",
        max_quarantine=f"{max_quarantine:.0%}",
    )


def run_fix(
    state: MeridianState,
    *,
    model_name: str,
    chat: ToolChat,
    tools: list[BaseTool],
    max_tool_calls: int,
    max_row_loss: float,
    max_quarantine: float,
    prompt_version: str = FIX_PROMPT_VERSION,
) -> dict[str, object]:
    """Propose one remediation for the single case and count the Fix execution."""
    if len(state.cases) != 1:
        raise ValueError("a fix branch handles one incident")
    key, case = next(iter(state.cases.items()))
    if case.hypothesis is None:
        raise ValueError("fix runs after a hypothesis")
    feedback = _feedback(case)
    hypothesis_id = case.hypothesis.id
    started = begin_fix(case)
    prompt = render_fix_prompt(
        state.model_copy(update={"cases": {key: started}}),
        feedback=feedback,
        max_row_loss=max_row_loss,
        max_quarantine=max_quarantine,
    )
    draft, llm_calls, tool_calls = _loop(
        prompt,
        chat=chat,
        tools=tools,
        model_name=model_name,
        prompt_version=prompt_version,
        max_tool_calls=max_tool_calls,
    )
    patch = _patch(draft, started.incident.id, hypothesis_id, model_name, prompt_version)
    updated = finish_fix(started, patch)
    return {"cases": {key: updated}, "llm_calls": llm_calls, "tool_calls": tool_calls}


def _feedback(case: IncidentCase) -> str:
    parts: list[str] = []
    if case.verdicts:
        latest = case.verdicts[-1]
        if not latest.approved:
            for item in latest.objections:
                parts.append(f"{item.kind.value}: {item.message}")
    if case.verification is not None and not case.verification.passed:
        for check in case.verification.checks:
            if not check.passed:
                parts.append(f"verifier {check.name}: {check.detail}")
    return "\n".join(parts)


def _loop(
    prompt: str,
    *,
    chat: ToolChat,
    tools: list[BaseTool],
    model_name: str,
    prompt_version: str,
    max_tool_calls: int,
) -> tuple[PatchDraft | None, list[LLMCall], list[ToolCall]]:
    tools_by_name = {tool.name: tool for tool in tools}
    bound = chat.bind_tools(tools)
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]
    llm_calls: list[LLMCall] = []
    tool_calls: list[ToolCall] = []
    steps = 0
    while steps < max_tool_calls:
        response, call = _invoke(bound, messages, model_name, prompt_version, len(llm_calls) + 1)
        llm_calls.append(call)
        requested = _first_tool_call(response) if call.succeeded else None
        if requested is None:
            break
        name, args, call_id = requested
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
            )
        )
        payload, record = _execute(name, args, tools_by_name)
        tool_calls.append(record)
        messages.append(ToolMessage(content=payload, tool_call_id=call_id, name=name))
        steps += 1
    structured = bind_structured(chat, PatchDraft)
    draft, call = _invoke_structured(
        structured,
        messages,
        model_name,
        prompt_version,
        len(llm_calls) + 1,
    )
    llm_calls.append(call)
    return draft, llm_calls, tool_calls


def _patch(
    draft: PatchDraft | None,
    incident_id: UUID,
    hypothesis_id: UUID,
    model_name: str,
    prompt_version: str,
) -> Patch:
    kind = RemediationKind.ESCALATE if draft is None else kind_or_escalate(draft.remediation_kind)
    source = escalate_source()
    test = "def check(df):\n    return None\n"
    if draft is not None and kind is not RemediationKind.ESCALATE and draft.module_source.strip():
        source = draft.module_source
    if draft is not None and draft.regression_test.strip():
        test = draft.regression_test
    path = f"remediations/{incident_id}.py"
    return Patch(
        id=patch_id_for(incident_id, prompt_version),
        incident_id=incident_id,
        hypothesis_id=hypothesis_id,
        remediation_kind=kind,
        diff=unified_diff(path, source),
        regression_test=test,
        files_touched=[path],
        author_model=model_name,
    )


def _execute(
    name: str,
    args: dict[str, object],
    tools_by_name: dict[str, BaseTool],
) -> tuple[str, ToolCall]:
    tool = tools_by_name.get(name)
    started = time.perf_counter()
    payload = ""
    ok = False
    if tool is None:
        payload = f"unknown tool {name}"
    else:
        try:
            payload = invoke_tool(tool, args)
            ok = True
        except Exception as exc:
            payload = str(exc)
    elapsed = max(int((time.perf_counter() - started) * 1000), 0)
    record = ToolCall(
        node=NODE,
        tool=name,
        args_hash=_args_hash(args),
        ref_count=len(collect_refs(payload)) if ok else 0,
        latency_ms=elapsed,
        ok=ok,
    )
    return payload, record


def _invoke(
    bound: Any,
    messages: list[BaseMessage],
    model_name: str,
    prompt_version: str,
    attempt: int,
) -> tuple[object, LLMCall]:
    started = time.perf_counter()
    try:
        response = bound.invoke(messages)
        succeeded = True
    except Exception:
        response = None
        succeeded = False
    return response, _call(response, model_name, prompt_version, attempt, started, succeeded)


def _invoke_structured(
    bound: Any,
    messages: list[BaseMessage],
    model_name: str,
    prompt_version: str,
    attempt: int,
) -> tuple[PatchDraft | None, LLMCall]:
    started = time.perf_counter()
    try:
        result = bound.invoke(messages)
    except Exception:
        result = None
    draft = _draft_from(result)
    raw = result.get("raw") if isinstance(result, dict) else result
    return draft, _call(raw, model_name, prompt_version, attempt, started, draft is not None)


def _draft_from(result: object) -> PatchDraft | None:
    if not isinstance(result, dict) or result.get("parsing_error") is not None:
        return None
    candidate = result.get("parsed")
    if isinstance(candidate, PatchDraft):
        return candidate
    if isinstance(candidate, dict):
        try:
            return PatchDraft.model_validate(candidate)
        except ValidationError:
            return None
    return None


def _first_tool_call(response: object) -> tuple[str, dict[str, object], str] | None:
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


def _call(
    raw: object,
    model_name: str,
    prompt_version: str,
    attempt: int,
    started: float,
    succeeded: bool,
) -> LLMCall:
    message = raw.get("raw") if isinstance(raw, dict) else raw
    usage = getattr(message, "usage_metadata", None)
    return LLMCall(
        node=NODE,
        model=model_name,
        prompt_version=prompt_version,
        input_tokens=_token(usage, "input_tokens"),
        output_tokens=_token(usage, "output_tokens"),
        latency_ms=max(int((time.perf_counter() - started) * 1000), 0),
        attempt=attempt,
        succeeded=succeeded,
    )


def _token(usage: object, key: str) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _args_hash(args: dict[str, object]) -> str:
    payload = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
