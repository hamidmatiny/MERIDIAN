"""Triage one incident at a time. The model may raise severity, never lower it."""

import time
from string import Template
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from meridian.agents.severity_rules import floor_for
from meridian.agents.structured import bind_structured
from meridian.contracts.incident import Incident
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.severity import Severity, more_severe
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageAssessment, TriageResult
from meridian.llm.errors import RunAborted, call_model
from meridian.llm.prompt_size import estimate_tokens, incident_json, tagged_human
from meridian.llm.submit import output_method_of, run_submit_only
from meridian.llm.usage import make_llm_call
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


def render_triage_prompt(incident: Incident, floor: Severity) -> tuple[str, str]:
    """Fill the versioned triage prompt for one incident."""
    downstream = ",".join(incident.blast_radius) if incident.blast_radius else "-"
    template = Template(load_prompt(TRIAGE_PROMPT_NAME, TRIAGE_PROMPT_VERSION))
    sent_json, full_json = incident_json(incident)
    sent = template.substitute(
        floor=floor.value,
        incident_id=str(incident.id),
        downstream=downstream,
        incident_json=sent_json,
    )
    uncut = template.substitute(
        floor=floor.value,
        incident_id=str(incident.id),
        downstream=downstream,
        incident_json=full_json,
    )
    return sent, uncut


def run_triage(
    state: MeridianState,
    *,
    model_name: str,
    chat: StructuredChat,
    prompt_version: str = TRIAGE_PROMPT_VERSION,
) -> dict[str, object]:
    """Triage the single case in this branch. Return only the new model calls."""
    if state.aborted:
        return {"aborted": True, "abort_reason": state.abort_reason}
    if len(state.cases) != 1:
        raise ValueError("a triage branch handles one incident")
    key, case = next(iter(state.cases.items()))
    try:
        result, attempts = _triage_one(
            case.incident,
            model_name=model_name,
            chat=chat,
            prompt_version=prompt_version,
        )
    except RunAborted as exc:
        return {"aborted": True, "abort_reason": str(exc)}
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
    sent, uncut = render_triage_prompt(incident, floor)
    messages: list[BaseMessage] = [
        tagged_human(sent, before=estimate_tokens(uncut), after=estimate_tokens(sent)),
    ]
    if output_method_of(chat) in {"submit_tool", "json_mode"}:
        parsed, calls = run_submit_only(
            chat,
            messages,
            TriageAssessment,
            tool_name="submit_triage_assessment",
            node="triage",
            model_name=model_name,
            prompt_version=prompt_version,
        )
        if isinstance(parsed, TriageAssessment) and parsed.incident_id == incident.id:
            return _accepted(parsed, incident, floor, prompt_version), calls
        return _degraded(incident, floor, prompt_version), calls
    bound = bind_structured(chat, TriageAssessment)
    attempts: list[LLMCall] = []
    for attempt in range(1, MAX_TRIAGE_ATTEMPTS + 1):
        parsed, call = _attempt(bound, messages, model_name, prompt_version, attempt)
        attempts.append(call)
        if parsed is None or parsed.incident_id != incident.id:
            if parsed is not None:
                attempts[-1] = call.model_copy(update={"succeeded": False})
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
        return result, attempts
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
    return degraded, attempts


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
    result, tokens_before, tokens_after = call_model(bound, messages)
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
    call = make_llm_call(
        node="triage",
        model=model_name,
        prompt_version=prompt_version,
        raw=raw,
        latency_ms=latency_ms,
        attempt=attempt,
        succeeded=succeeded,
        bound=bound,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        structured=True,
    )
    return parsed, call


def _accepted(
    parsed: TriageAssessment,
    incident: Incident,
    floor: Severity,
    prompt_version: str,
) -> TriageResult:
    final = more_severe(parsed.severity, floor)
    return TriageResult(
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


def _degraded(incident: Incident, floor: Severity, prompt_version: str) -> TriageResult:
    return TriageResult(
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
