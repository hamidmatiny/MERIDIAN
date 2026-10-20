"""Triage one incident at a time. The model may raise severity, never lower it."""

import time
from string import Template
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from meridian.agents.severity_rules import floor_for
from meridian.agents.structured import bind_structured
from meridian.contracts.incident import Incident
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.severity import Severity, more_severe
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageAssessment, TriageResult
from meridian.prompts import TRIAGE_PROMPT_NAME, TRIAGE_PROMPT_VERSION, load_prompt

MAX_TRIAGE_ATTEMPTS = 2
_DEGRADED_RATIONALE = "Structured output failed. Severity stayed at the deterministic floor."


class StructuredChat(Protocol):
    """Anything that can request a schema the way ChatAnthropic does."""

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any: ...


def render_triage_prompt(incident: Incident, floor: Severity) -> str:
    """Fill the versioned triage prompt for one incident."""
    downstream = ",".join(incident.blast_radius) if incident.blast_radius else "-"
    template = Template(load_prompt(TRIAGE_PROMPT_NAME, TRIAGE_PROMPT_VERSION))
    return template.substitute(
        floor=floor.value,
        incident_id=str(incident.id),
        downstream=downstream,
        incident_json=incident.model_dump_json(),
    )


def run_triage(
    state: MeridianState,
    *,
    model_name: str,
    chat: StructuredChat,
    prompt_version: str = TRIAGE_PROMPT_VERSION,
) -> dict[str, object]:
    """Triage the single case in this branch. Return only the new model calls."""
    if len(state.cases) != 1:
        raise ValueError("a triage branch handles one incident")
    key, case = next(iter(state.cases.items()))
    result, attempts = _triage_one(
        case.incident,
        model_name=model_name,
        chat=chat,
        prompt_version=prompt_version,
    )
    updated = case.model_copy(update={"triage": result})
    return {"cases": {key: updated}, "llm_calls": attempts}


def _triage_one(
    incident: Incident,
    *,
    model_name: str,
    chat: StructuredChat,
    prompt_version: str,
) -> tuple[TriageResult, list[LLMCall]]:
    floor = floor_for(incident.fault_type, len(incident.blast_radius))
    prompt = render_triage_prompt(incident, floor)
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]
    bound = bind_structured(chat, TriageAssessment)
    calls: list[LLMCall] = []
    for attempt in range(1, MAX_TRIAGE_ATTEMPTS + 1):
        parsed, call = _attempt(bound, messages, model_name, prompt_version, attempt)
        calls.append(call)
        if parsed is None or parsed.incident_id != incident.id:
            if parsed is not None:
                calls[-1] = call.model_copy(update={"succeeded": False})
            continue
        final = more_severe(parsed.severity, floor)
        result = TriageResult(
            incident_id=incident.id,
            severity=final,
            rationale=parsed.rationale,
            affected_consumers=list(parsed.affected_consumers),
            confidence=parsed.confidence,
            llm_severity=parsed.severity,
            floor_severity=floor,
            final_severity=final,
            prompt_version=prompt_version,
            triage_degraded=False,
        )
        return result, calls
    degraded = TriageResult(
        incident_id=incident.id,
        severity=floor,
        rationale=_DEGRADED_RATIONALE,
        affected_consumers=list(incident.blast_radius),
        confidence=0.0,
        llm_severity=None,
        floor_severity=floor,
        final_severity=floor,
        prompt_version=prompt_version,
        triage_degraded=True,
    )
    return degraded, calls


def _attempt(
    bound: Any,
    messages: list[BaseMessage],
    model_name: str,
    prompt_version: str,
    attempt: int,
) -> tuple[TriageAssessment | None, LLMCall]:
    started = time.perf_counter()
    parsed: TriageAssessment | None = None
    raw: object = None
    succeeded = False
    try:
        result = bound.invoke(messages)
    except Exception:
        result = None
    latency_ms = int((time.perf_counter() - started) * 1000)
    if isinstance(result, dict):
        raw = result.get("raw")
        candidate = result.get("parsed")
        error = result.get("parsing_error")
        if error is None and isinstance(candidate, dict):
            try:
                candidate = TriageAssessment.model_validate(candidate)
            except ValidationError:
                candidate = None
        if error is None and isinstance(candidate, TriageAssessment):
            parsed = candidate
            succeeded = True
    input_tokens, output_tokens = _usage(raw)
    call = LLMCall(
        node="triage",
        model=model_name,
        prompt_version=prompt_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        attempt=attempt,
        succeeded=succeeded,
    )
    return parsed, call


def _usage(raw: object) -> tuple[int, int]:
    usage = getattr(raw, "usage_metadata", None)
    if not isinstance(usage, dict):
        return 0, 0
    return _token(usage.get("input_tokens")), _token(usage.get("output_tokens"))


def _token(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)
