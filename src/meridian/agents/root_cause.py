"""Bounded root-cause loop. Tools first, then a grounded hypothesis."""

import hashlib
import json
import time
from pathlib import Path
from string import Template
from typing import Any, Protocol
from uuid import UUID, uuid5

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meridian.contracts.hypothesis import Evidence, Hypothesis
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.namespace import MERIDIAN_NAMESPACE
from meridian.contracts.state import MeridianState
from meridian.contracts.tool_call import ToolCall
from meridian.mcp_servers.client import invoke_tool
from meridian.mcp_servers.impl import publish_incident
from meridian.prompts import ROOT_CAUSE_PROMPT_NAME, ROOT_CAUSE_PROMPT_VERSION, load_prompt

NODE = "root_cause"
CONFIDENCE_DROP = 0.25
_FALLBACK_SUMMARY = "Tool budget exhausted before a grounded hypothesis."
_FALLBACK_MECHANISM = "No mechanism was established within the tool budget."


class ToolChat(Protocol):
    """A chat model that can bind tools and emit a schema."""

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> Any: ...

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any: ...


class RootCauseDraft(BaseModel):
    """Fields the model fills. The node assigns id, incident id, and violations."""

    model_config = ConfigDict(extra="ignore")

    root_cause_summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
    causal_event_id: str | None = None
    suspect_columns: list[str] = Field(default_factory=list)
    mechanism: str = "unspecified"


class LoopTurn(BaseModel):
    """One message in the tool loop, stored without LangChain objects."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    content: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    tool_args_json: str = "{}"


class LoopState(BaseModel):
    """Private state of the tool-loop subgraph."""

    model_config = ConfigDict(extra="forbid")

    incident_id: UUID
    prompt_version: str
    turns: list[LoopTurn]
    steps: int = 0
    refs: list[str] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    hypothesis: Hypothesis | None = None


def render_root_cause_prompt(state: MeridianState, *, max_tool_calls: int) -> str:
    """Fill the versioned root-cause prompt for the single case in this branch."""
    _key, case = next(iter(state.cases.items()))
    incident = case.incident
    downstream = ",".join(incident.blast_radius) if incident.blast_radius else "-"
    template = Template(load_prompt(ROOT_CAUSE_PROMPT_NAME, ROOT_CAUSE_PROMPT_VERSION))
    return template.substitute(
        incident_json=incident.model_dump_json(),
        fault_type=incident.fault_type.value,
        dataset=incident.dataset,
        downstream=downstream,
        max_tool_calls=str(max_tool_calls),
    )


def run_root_cause(
    state: MeridianState,
    *,
    model_name: str,
    chat: ToolChat,
    tools: list[BaseTool],
    max_tool_calls: int,
    evidence_root: Path,
    prompt_version: str = ROOT_CAUSE_PROMPT_VERSION,
) -> dict[str, object]:
    """Investigate one triaged case and return its hypothesis plus new calls."""
    if len(state.cases) != 1:
        raise ValueError("a root-cause branch handles one incident")
    key, case = next(iter(state.cases.items()))
    if case.triage is None:
        raise ValueError("root cause runs after triage")
    publish_incident(evidence_root, case.incident)
    prompt = render_root_cause_prompt(state, max_tool_calls=max_tool_calls)
    loop = _compile_loop(
        chat=chat,
        tools=tools,
        model_name=model_name,
        prompt_version=prompt_version,
        max_tool_calls=max_tool_calls,
    )
    result = loop.invoke(
        LoopState(
            incident_id=case.incident.id,
            prompt_version=prompt_version,
            turns=[LoopTurn(kind="human", content=prompt)],
        )
    )
    finished = result if isinstance(result, LoopState) else LoopState.model_validate(result)
    hypothesis = finished.hypothesis
    if hypothesis is None:
        hypothesis = _fallback(case.incident.id, prompt_version)
    hypothesis = hypothesis.model_copy(
        update={
            "id": uuid5(MERIDIAN_NAMESPACE, f"hypothesis|{case.incident.id}|{prompt_version}"),
            "incident_id": case.incident.id,
        }
    )
    updated = case.model_copy(update={"hypothesis": hypothesis})
    return {
        "cases": {key: updated},
        "llm_calls": list(finished.llm_calls),
        "tool_calls": list(finished.tool_calls),
    }


def ground(hypothesis: Hypothesis, refs: set[str]) -> Hypothesis:
    """Drop citations the tools did not return. Never add a reference."""
    kept: list[Evidence] = []
    violations = 0
    for item in hypothesis.evidence:
        if item.reference in refs:
            kept.append(item)
        else:
            violations += 1
    causal = hypothesis.causal_event_id
    if causal is not None and f"event:{causal}" not in refs:
        causal = None
        violations += 1
    confidence = max(0.0, hypothesis.confidence - CONFIDENCE_DROP * violations)
    return hypothesis.model_copy(
        update={
            "evidence": kept,
            "causal_event_id": causal,
            "confidence": confidence,
            "grounding_violations": violations,
        }
    )


def collect_refs(payload: str) -> list[str]:
    """Every string stored under a ``ref`` key in a tool payload."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    found: list[str] = []
    _walk_refs(data, found)
    return found


def _compile_loop(
    *,
    chat: ToolChat,
    tools: list[BaseTool],
    model_name: str,
    prompt_version: str,
    max_tool_calls: int,
) -> Any:
    tools_by_name = {tool.name: tool for tool in tools}
    bound = chat.bind_tools(tools)

    def plan(state: LoopState) -> dict[str, object]:
        if state.steps >= max_tool_calls:
            return _force_answer(state, chat, model_name, prompt_version)
        response, call = _invoke_model(
            bound,
            _messages(state.turns),
            state,
            model_name,
            prompt_version,
        )
        calls = [*state.llm_calls, call]
        requested = _first_tool_call(response) if call.succeeded else None
        if requested is None:
            return _force_answer(state, chat, model_name, prompt_version, prior=calls)
        name, args, call_id = requested
        turn = LoopTurn(
            kind="ai",
            content="",
            tool_name=name,
            tool_call_id=call_id,
            tool_args_json=json.dumps(args, sort_keys=True, default=str),
        )
        return {"turns": [*state.turns, turn], "llm_calls": calls}

    def act(state: LoopState) -> dict[str, object]:
        return _execute(state, tools_by_name)

    def finish(state: LoopState) -> dict[str, object]:
        current = state.hypothesis or _fallback(state.incident_id, state.prompt_version)
        return {"hypothesis": ground(current, set(state.refs))}

    graph: StateGraph[LoopState, None, LoopState, LoopState] = StateGraph(LoopState)
    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_node("finish", finish)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _route, ["act", "finish"])
    graph.add_edge("act", "plan")
    graph.add_edge("finish", END)
    return graph.compile()


def _route(state: LoopState) -> str:
    if state.hypothesis is not None:
        return "finish"
    return "act"


def _force_answer(
    state: LoopState,
    chat: ToolChat,
    model_name: str,
    prompt_version: str,
    prior: list[LLMCall] | None = None,
) -> dict[str, object]:
    structured = chat.with_structured_output(RootCauseDraft, include_raw=True)
    parsed, call = _invoke_structured(
        structured,
        _messages(state.turns),
        state.incident_id,
        model_name,
        prompt_version,
        attempt=(len(prior) if prior is not None else len(state.llm_calls)) + 1,
    )
    calls = [*(prior if prior is not None else state.llm_calls), call]
    hypothesis = parsed if parsed is not None else _fallback(state.incident_id, prompt_version)
    return {"hypothesis": hypothesis, "llm_calls": calls}


def _execute(state: LoopState, tools_by_name: dict[str, BaseTool]) -> dict[str, object]:
    last = state.turns[-1]
    args = json.loads(last.tool_args_json)
    if not isinstance(args, dict):
        args = {}
    tool = tools_by_name.get(last.tool_name)
    started = time.perf_counter()
    payload = ""
    ok = False
    if tool is None:
        payload = f"unknown tool {last.tool_name}"
    else:
        try:
            payload = invoke_tool(tool, args)
            ok = True
        except Exception as exc:
            payload = str(exc)
    latency_ms = max(int((time.perf_counter() - started) * 1000), 0)
    refs = collect_refs(payload) if ok else []
    record = ToolCall(
        node=NODE,
        tool=last.tool_name or "unknown",
        args_hash=_args_hash(args),
        ref_count=len(refs),
        latency_ms=latency_ms,
        ok=ok,
    )
    turn = LoopTurn(
        kind="tool",
        content=payload,
        tool_name=last.tool_name,
        tool_call_id=last.tool_call_id,
    )
    return {
        "turns": [*state.turns, turn],
        "tool_calls": [*state.tool_calls, record],
        "refs": _union(state.refs, refs),
        "steps": state.steps + 1,
    }


def _invoke_model(
    bound: Any,
    messages: list[BaseMessage],
    state: LoopState,
    model_name: str,
    prompt_version: str,
) -> tuple[object, LLMCall]:
    started = time.perf_counter()
    try:
        response = bound.invoke(messages)
        succeeded = True
    except Exception:
        response = None
        succeeded = False
    call = _call_record(
        response,
        model_name,
        prompt_version,
        len(state.llm_calls) + 1,
        started,
        succeeded,
    )
    return response, call


def _invoke_structured(
    bound: Any,
    messages: list[BaseMessage],
    incident_id: UUID,
    model_name: str,
    prompt_version: str,
    attempt: int,
) -> tuple[Hypothesis | None, LLMCall]:
    started = time.perf_counter()
    try:
        result = bound.invoke(messages)
    except Exception:
        result = None
    draft = _draft_from(result)
    hypothesis = None if draft is None else _from_draft(draft, incident_id, prompt_version)
    raw = result.get("raw") if isinstance(result, dict) else result
    return hypothesis, _call_record(
        raw,
        model_name,
        prompt_version,
        attempt,
        started,
        hypothesis is not None,
    )


def _from_draft(draft: RootCauseDraft, incident_id: UUID, prompt_version: str) -> Hypothesis | None:
    """Build a hypothesis. An unparseable causal id fails grounding instead of being kept."""
    try:
        mechanism = draft.mechanism.strip() or "unspecified"
        return Hypothesis(
            id=uuid5(MERIDIAN_NAMESPACE, f"hypothesis|{incident_id}|{prompt_version}"),
            incident_id=incident_id,
            root_cause_summary=draft.root_cause_summary,
            evidence=list(draft.evidence),
            confidence=draft.confidence,
            causal_event_id=_causal_for_grounding(draft.causal_event_id),
            suspect_columns=list(draft.suspect_columns),
            mechanism=mechanism,
            grounding_violations=0,
        )
    except (ValidationError, ValueError):
        return None


def _draft_from(result: object) -> RootCauseDraft | None:
    if not isinstance(result, dict) or result.get("parsing_error") is not None:
        return None
    candidate = result.get("parsed")
    if isinstance(candidate, RootCauseDraft):
        return candidate
    if isinstance(candidate, dict):
        try:
            return RootCauseDraft.model_validate(candidate)
        except ValidationError:
            return None
    return None


def _fallback(incident_id: UUID, prompt_version: str) -> Hypothesis:
    return Hypothesis(
        id=uuid5(MERIDIAN_NAMESPACE, f"hypothesis|{incident_id}|{prompt_version}"),
        incident_id=incident_id,
        root_cause_summary=_FALLBACK_SUMMARY,
        evidence=[],
        confidence=0.0,
        causal_event_id=None,
        suspect_columns=[],
        mechanism=_FALLBACK_MECHANISM,
        grounding_violations=0,
    )


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


def _messages(turns: list[LoopTurn]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in turns:
        if turn.kind == "human":
            messages.append(HumanMessage(content=turn.content))
        elif turn.kind == "tool":
            messages.append(
                ToolMessage(
                    content=turn.content,
                    tool_call_id=turn.tool_call_id or "tool-call",
                    name=turn.tool_name or None,
                )
            )
        elif turn.tool_name:
            messages.append(
                AIMessage(
                    content=turn.content,
                    tool_calls=[
                        {
                            "name": turn.tool_name,
                            "args": json.loads(turn.tool_args_json),
                            "id": turn.tool_call_id or "tool-call",
                            "type": "tool_call",
                        }
                    ],
                )
            )
        else:
            messages.append(AIMessage(content=turn.content))
    return messages


def _call_record(
    raw: object,
    model_name: str,
    prompt_version: str,
    attempt: int,
    started: float,
    succeeded: bool,
) -> LLMCall:
    message = raw.get("raw") if isinstance(raw, dict) else raw
    usage = getattr(message, "usage_metadata", None)
    input_tokens, output_tokens = _usage(usage)
    return LLMCall(
        node=NODE,
        model=model_name,
        prompt_version=prompt_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=max(int((time.perf_counter() - started) * 1000), 0),
        attempt=attempt,
        succeeded=succeeded,
    )


def _usage(usage: object) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    return _token(usage.get("input_tokens")), _token(usage.get("output_tokens"))


def _token(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _causal_for_grounding(value: str | None) -> UUID | None:
    """Parse a causal id. An unparseable id becomes one no tool could have returned."""
    if value is None or value.strip() == "":
        return None
    try:
        return UUID(value)
    except ValueError:
        return uuid5(MERIDIAN_NAMESPACE, f"unparseable-causal|{value}")


def _args_hash(args: dict[str, object]) -> str:
    payload = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _union(left: list[str], right: list[str]) -> list[str]:
    seen = list(left)
    for item in right:
        if item not in seen:
            seen.append(item)
    return seen


def _walk_refs(node: object, found: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ref" and isinstance(value, str):
                found.append(value)
            else:
                _walk_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_refs(item, found)
