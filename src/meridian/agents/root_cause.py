"""Bounded root-cause loop. Tools first, then a grounded hypothesis."""

import hashlib
import json
import time
from pathlib import Path
from string import Template
from typing import Any, Protocol
from uuid import UUID, uuid5

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meridian.agents.structured import bind_structured
from meridian.contracts.grounding import (
    GroundingViolation,
    canonical_event_ref,
    event_uuid,
    ref_sample,
)
from meridian.contracts.hypothesis import Evidence, Hypothesis
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.namespace import MERIDIAN_NAMESPACE
from meridian.contracts.state import MeridianState
from meridian.contracts.tool_call import ToolCall
from meridian.llm.errors import RunAborted, call_model, unwrap_abort
from meridian.llm.prompt_size import estimate_tokens, incident_json, tagged_human
from meridian.llm.submit import output_method_of, run_submit_only, run_tool_loop
from meridian.llm.usage import make_llm_call
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
    tokens_before: int = 0
    tokens_after: int = 0


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
    causal_raw: str = ""
    budget_fallback: bool = False


def render_root_cause_prompt(state: MeridianState, *, max_tool_calls: int) -> tuple[str, str]:
    """Fill the versioned root-cause prompt for the single case in this branch."""
    _key, case = next(iter(state.cases.items()))
    incident = case.incident
    downstream = ",".join(incident.blast_radius) if incident.blast_radius else "-"
    sent_json, full_json = incident_json(incident)
    template = Template(load_prompt(ROOT_CAUSE_PROMPT_NAME, ROOT_CAUSE_PROMPT_VERSION))
    fields = {
        "fault_type": incident.fault_type.value,
        "dataset": incident.dataset,
        "downstream": downstream,
        "max_tool_calls": str(max_tool_calls),
    }
    sent = template.substitute(incident_json=sent_json, **fields)
    uncut = template.substitute(incident_json=full_json, **fields)
    return sent, uncut


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
    if state.aborted:
        return {"aborted": True, "abort_reason": state.abort_reason}
    if len(state.cases) != 1:
        raise ValueError("a root-cause branch handles one incident")
    key, case = next(iter(state.cases.items()))
    if case.triage is None:
        raise ValueError("root cause runs after triage")
    publish_incident(evidence_root, case.incident)
    sent, uncut = render_root_cause_prompt(state, max_tool_calls=max_tool_calls)
    if output_method_of(chat) == "submit_tool":
        try:
            hypothesis, llm_calls, tool_calls, budget_fallback = _submit_root(
                sent,
                uncut,
                chat=chat,
                tools=tools,
                model_name=model_name,
                prompt_version=prompt_version,
                max_tool_calls=max_tool_calls,
                incident_id=case.incident.id,
            )
        except RunAborted as exc:
            return {"aborted": True, "abort_reason": str(exc)}
        updated = case.model_copy(
            update={
                "hypothesis": hypothesis,
                "hypothesis_from_budget_fallback": budget_fallback,
            }
        )
        return {
            "cases": {key: updated},
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
        }
    loop = _compile_loop(
        chat=chat,
        tools=tools,
        model_name=model_name,
        prompt_version=prompt_version,
        max_tool_calls=max_tool_calls,
    )
    try:
        result = loop.invoke(
            LoopState(
                incident_id=case.incident.id,
                prompt_version=prompt_version,
                turns=[
                    LoopTurn(
                        kind="human",
                        content=sent,
                        tokens_before=estimate_tokens(uncut),
                        tokens_after=estimate_tokens(sent),
                    )
                ],
            )
        )
    except Exception as exc:
        aborted = exc if isinstance(exc, RunAborted) else unwrap_abort(exc)
        if aborted is None:
            raise
        return {"aborted": True, "abort_reason": str(aborted)}
    finished = result if isinstance(result, LoopState) else LoopState.model_validate(result)
    resolved = finished.hypothesis
    budget_fallback = finished.budget_fallback
    if resolved is None:
        resolved = _fallback(case.incident.id, prompt_version)
        budget_fallback = True
    hypothesis = resolved.model_copy(
        update={
            "id": uuid5(MERIDIAN_NAMESPACE, f"hypothesis|{case.incident.id}|{prompt_version}"),
            "incident_id": case.incident.id,
        }
    )
    updated = case.model_copy(
        update={
            "hypothesis": hypothesis,
            "hypothesis_from_budget_fallback": budget_fallback,
        }
    )
    return {
        "cases": {key: updated},
        "llm_calls": list(finished.llm_calls),
        "tool_calls": list(finished.tool_calls),
    }


def ground(hypothesis: Hypothesis, refs: set[str], *, causal_raw: str = "") -> Hypothesis:
    """Drop citations the tools did not return. Never add a reference.

    ``event:<id>`` and ``<id>`` name the same event. An id that appears anywhere in
    this case's tool output is grounded; a later ``get_event`` is not required.
    """
    findings: list[GroundingViolation] = []
    kept: list[Evidence] = []
    for item in hypothesis.evidence:
        if _cited(item.reference, refs):
            kept.append(item)
        else:
            findings.append(_finding("evidence", item.reference, refs))
    raw = causal_raw.strip()
    if raw == "" and hypothesis.causal_event_id is not None:
        raw = str(hypothesis.causal_event_id)
    causal = event_uuid(raw) if raw else hypothesis.causal_event_id
    if raw and not _cited(raw, refs):
        findings.append(_finding("causal_event_id", raw, refs))
        causal = None
    confidence = max(0.0, hypothesis.confidence - CONFIDENCE_DROP * len(findings))
    return hypothesis.model_copy(
        update={
            "evidence": kept,
            "causal_event_id": causal,
            "confidence": confidence,
            "grounding_violations": len(findings),
            "grounding_findings": findings,
        }
    )


def collect_refs(payload: str) -> list[str]:
    """Refs and event ids in a tool payload.

    A string under ``ref`` is kept as written. Any ``event:<uuid>`` or bare UUID is
    also stored in both forms, so an id from ``list_events`` grounds a later citation.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    found: list[str] = []
    _walk_refs(data, found)
    return found


def _cited(value: str, refs: set[str]) -> bool:
    if value in refs:
        return True
    canonical = canonical_event_ref(value)
    if canonical is None:
        return False
    key = canonical.removeprefix("event:")
    return canonical in refs or key in refs


def _finding(field: str, raw: str, refs: set[str]) -> GroundingViolation:
    text = raw.strip() or "(empty)"
    normalized = canonical_event_ref(text) or text
    return GroundingViolation(
        field=field,
        raw=text,
        normalized=normalized,
        allowed_refs_sample=ref_sample(refs),
    )


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
        return {"hypothesis": ground(current, set(state.refs), causal_raw=state.causal_raw)}

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
    if output_method_of(chat) == "json_mode":
        return _json_answer(state, chat, model_name, prompt_version, prior)
    structured = bind_structured(chat, RootCauseDraft)
    parsed, causal_raw, call = _invoke_structured(
        structured,
        _messages(state.turns),
        state.incident_id,
        model_name,
        prompt_version,
        attempt=(len(prior) if prior is not None else len(state.llm_calls)) + 1,
    )
    calls = [*(prior if prior is not None else state.llm_calls), call]
    if parsed is None:
        return {
            "hypothesis": _fallback(state.incident_id, prompt_version),
            "llm_calls": calls,
            "causal_raw": causal_raw,
            "budget_fallback": True,
        }
    return {
        "hypothesis": parsed,
        "llm_calls": calls,
        "causal_raw": causal_raw,
        "budget_fallback": False,
    }


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
    response, tokens_before, tokens_after = call_model(bound, messages)
    succeeded = response is not None
    call = _call_record(
        response,
        model_name,
        prompt_version,
        len(state.llm_calls) + 1,
        started,
        succeeded,
        bound,
        tokens_before,
        tokens_after,
    )
    return response, call


def _submit_root(
    sent: str,
    uncut: str,
    *,
    chat: ToolChat,
    tools: list[BaseTool],
    model_name: str,
    prompt_version: str,
    max_tool_calls: int,
    incident_id: UUID,
) -> tuple[Hypothesis, list[LLMCall], list[ToolCall], bool]:
    """Investigate with a submit tool. A missing submit becomes the budget fallback."""
    human = tagged_human(sent, before=estimate_tokens(uncut), after=estimate_tokens(sent))
    parsed, calls, records, payloads = run_tool_loop(
        chat,
        [human],
        RootCauseDraft,
        tool_name="submit_hypothesis",
        mcp_tools=tools,
        node=NODE,
        model_name=model_name,
        prompt_version=prompt_version,
        max_tool_calls=max_tool_calls,
    )
    refs: list[str] = []
    updated_records: list[ToolCall] = []
    for record, payload in zip(records, payloads, strict=True):
        found = collect_refs(payload)
        refs = _union(refs, found)
        updated_records.append(record.model_copy(update={"ref_count": len(found)}))
    draft = parsed if isinstance(parsed, RootCauseDraft) else None
    built = None if draft is None else _from_draft(draft, incident_id, prompt_version)
    if built is None:
        return _fallback(incident_id, prompt_version), calls, updated_records, True
    hypothesis = ground(built[0], set(refs), causal_raw=built[1])
    return hypothesis, calls, updated_records, False


def _json_answer(
    state: LoopState,
    chat: ToolChat,
    model_name: str,
    prompt_version: str,
    prior: list[LLMCall] | None,
) -> dict[str, object]:
    parsed, extra = run_submit_only(
        chat,
        _messages(state.turns),
        RootCauseDraft,
        tool_name="submit_hypothesis",
        node=NODE,
        model_name=model_name,
        prompt_version=prompt_version,
    )
    calls = [*(prior if prior is not None else state.llm_calls), *extra]
    draft = parsed if isinstance(parsed, RootCauseDraft) else None
    built = None if draft is None else _from_draft(draft, state.incident_id, prompt_version)
    if built is None:
        return {
            "hypothesis": _fallback(state.incident_id, prompt_version),
            "llm_calls": calls,
            "causal_raw": "",
            "budget_fallback": True,
        }
    return {
        "hypothesis": built[0],
        "llm_calls": calls,
        "causal_raw": built[1],
        "budget_fallback": False,
    }


def _invoke_structured(
    bound: Any,
    messages: list[BaseMessage],
    incident_id: UUID,
    model_name: str,
    prompt_version: str,
    attempt: int,
) -> tuple[Hypothesis | None, str, LLMCall]:
    started = time.perf_counter()
    result, tokens_before, tokens_after = call_model(bound, messages)
    draft = _draft_from(result)
    built = None if draft is None else _from_draft(draft, incident_id, prompt_version)
    hypothesis = None if built is None else built[0]
    causal_raw = "" if built is None else built[1]
    raw = result.get("raw") if isinstance(result, dict) else result
    return (
        hypothesis,
        causal_raw,
        _call_record(
            raw,
            model_name,
            prompt_version,
            attempt,
            started,
            hypothesis is not None,
            bound,
            tokens_before,
            tokens_after,
            structured=True,
        ),
    )


def _from_draft(
    draft: RootCauseDraft,
    incident_id: UUID,
    prompt_version: str,
) -> tuple[Hypothesis, str] | None:
    """Build a hypothesis. The raw causal string is grounded later, not rewritten here."""
    try:
        mechanism = draft.mechanism.strip() or "unspecified"
        causal_raw = (draft.causal_event_id or "").strip()
        hypothesis = Hypothesis(
            id=uuid5(MERIDIAN_NAMESPACE, f"hypothesis|{incident_id}|{prompt_version}"),
            incident_id=incident_id,
            root_cause_summary=draft.root_cause_summary,
            evidence=list(draft.evidence),
            confidence=draft.confidence,
            causal_event_id=event_uuid(causal_raw),
            suspect_columns=list(draft.suspect_columns),
            mechanism=mechanism,
            grounding_violations=0,
        )
    except (ValidationError, ValueError):
        return None
    return hypothesis, causal_raw


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
            messages.append(
                tagged_human(
                    turn.content,
                    before=turn.tokens_before,
                    after=turn.tokens_after,
                )
            )
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
    bound: object,
    tokens_before: int,
    tokens_after: int,
    structured: bool = False,
) -> LLMCall:
    message = raw.get("raw") if isinstance(raw, dict) else raw
    return make_llm_call(
        node=NODE,
        model=model_name,
        prompt_version=prompt_version,
        raw=message,
        latency_ms=max(int((time.perf_counter() - started) * 1000), 0),
        attempt=attempt,
        succeeded=succeeded,
        bound=bound,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        structured=structured,
    )


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
            if isinstance(value, str) and (key == "ref" or canonical_event_ref(value) is not None):
                _remember(found, value)
            else:
                _walk_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str) and canonical_event_ref(item) is not None:
                _remember(found, item)
            else:
                _walk_refs(item, found)


def _remember(found: list[str], value: str) -> None:
    if value not in found:
        found.append(value)
    canonical = canonical_event_ref(value)
    if canonical is None:
        return
    bare = canonical.removeprefix("event:")
    if bare not in found:
        found.append(bare)
    if canonical not in found:
        found.append(canonical)
