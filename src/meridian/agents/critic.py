"""Critic. Grok reviews a patch and never sees the injected fault."""

import time
from string import Template
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meridian.contracts.llm_call import LLMCall
from meridian.contracts.remediation import Objection, ObjectionKind
from meridian.contracts.state import MeridianState
from meridian.contracts.verdict import Verdict
from meridian.graph.limits import objection, record_verdict
from meridian.prompts import CRITIC_PROMPT_NAME, CRITIC_PROMPT_VERSION, load_prompt

NODE = "critic"


class CriticChat(Protocol):
    """A chat model that emits a verdict draft."""

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any: ...


class VerdictDraft(BaseModel):
    """Fields the critic fills. The node assigns the patch id and the round."""

    model_config = ConfigDict(extra="ignore")

    approved: bool
    objections: list[dict[str, str]] = Field(default_factory=list)


def render_critic_prompt(state: MeridianState) -> str:
    """Fill the critic prompt from the incident, the hypothesis, and the patch."""
    _key, case = next(iter(state.cases.items()))
    hypothesis = "{}" if case.hypothesis is None else case.hypothesis.model_dump_json()
    patch = case.patch
    template = Template(load_prompt(CRITIC_PROMPT_NAME, CRITIC_PROMPT_VERSION))
    return template.substitute(
        incident_json=case.incident.model_dump_json(),
        hypothesis_json=hypothesis,
        remediation_kind=patch.remediation_kind.value if patch is not None else "ESCALATE",
        diff=patch.diff if patch is not None else "",
        regression_test=patch.regression_test if patch is not None else "",
    )


def run_critic(
    state: MeridianState,
    *,
    model_name: str,
    chat: CriticChat,
    prompt_version: str = CRITIC_PROMPT_VERSION,
) -> dict[str, object]:
    """Review the current patch. A rejection on round 3 escalates."""
    if len(state.cases) != 1:
        raise ValueError("a critic branch handles one incident")
    key, case = next(iter(state.cases.items()))
    if case.patch is None:
        raise ValueError("critic runs after a patch")
    prompt = render_critic_prompt(state)
    structured = chat.with_structured_output(VerdictDraft, include_raw=True)
    draft, call = _invoke(structured, [HumanMessage(content=prompt)], model_name, prompt_version)
    verdict = _verdict(draft, case.patch.id, model_name, case.critic_rounds + 1)
    updated = record_verdict(case, verdict)
    return {"cases": {key: updated}, "llm_calls": [call]}


def _verdict(
    draft: VerdictDraft | None,
    patch_id: object,
    model_name: str,
    round_number: int,
) -> Verdict:
    from uuid import UUID

    if not isinstance(patch_id, UUID):
        raise TypeError("patch id must be a UUID")
    if draft is None or not draft.approved:
        objections = _objections(None if draft is None else draft.objections)
        if not objections:
            objections = [objection(ObjectionKind.OTHER, "The review did not approve the patch.")]
        return Verdict(
            patch_id=patch_id,
            reviewer_model=model_name,
            approved=False,
            objections=objections,
            round=round_number,
        )
    return Verdict(
        patch_id=patch_id,
        reviewer_model=model_name,
        approved=True,
        objections=[],
        round=round_number,
    )


def _objections(raw: list[dict[str, str]] | None) -> list[Objection]:
    if raw is None:
        return []
    found: list[Objection] = []
    for item in raw:
        kind_text = item.get("kind", ObjectionKind.OTHER.value)
        message = item.get("message", "").strip()
        if message == "":
            continue
        try:
            kind = ObjectionKind(kind_text)
        except ValueError:
            kind = ObjectionKind.OTHER
        found.append(objection(kind, message))
    return found


def _invoke(
    bound: Any,
    messages: list[BaseMessage],
    model_name: str,
    prompt_version: str,
) -> tuple[VerdictDraft | None, LLMCall]:
    started = time.perf_counter()
    try:
        result = bound.invoke(messages)
    except Exception:
        result = None
    draft = _draft_from(result)
    raw = result.get("raw") if isinstance(result, dict) else result
    usage = getattr(raw, "usage_metadata", None)
    call = LLMCall(
        node=NODE,
        model=model_name,
        prompt_version=prompt_version,
        input_tokens=_token(usage, "input_tokens"),
        output_tokens=_token(usage, "output_tokens"),
        latency_ms=max(int((time.perf_counter() - started) * 1000), 0),
        attempt=1,
        succeeded=draft is not None,
    )
    return draft, call


def _draft_from(result: object) -> VerdictDraft | None:
    if not isinstance(result, dict) or result.get("parsing_error") is not None:
        return None
    candidate = result.get("parsed")
    if isinstance(candidate, VerdictDraft):
        return candidate
    if isinstance(candidate, dict):
        try:
            return VerdictDraft.model_validate(candidate)
        except ValidationError:
            return None
    return None


def _token(usage: object, key: str) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)
