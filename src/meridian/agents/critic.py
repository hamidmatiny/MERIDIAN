"""Critic. Grok reviews a patch and never sees the injected fault."""

import time
from string import Template
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meridian.agents.structured import bind_structured
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.remediation import Objection, ObjectionKind
from meridian.contracts.state import MeridianState
from meridian.contracts.verdict import Verdict
from meridian.graph.limits import objection, record_verdict
from meridian.llm.errors import RunAborted, call_model
from meridian.llm.prompt_size import estimate_tokens, incident_json, tagged_human
from meridian.llm.submit import CRITIC_CORRECTIVE, output_method_of, run_submit_only
from meridian.llm.usage import assemble_calls
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


def render_critic_prompt(state: MeridianState) -> tuple[str, str]:
    """Fill the critic prompt from the incident, the hypothesis, the patch, and the dry-run."""
    _key, case = next(iter(state.cases.items()))
    hypothesis = "{}" if case.hypothesis is None else case.hypothesis.model_dump_json()
    patch = case.patch
    dry_run = "{}" if case.dry_run is None else case.dry_run.model_dump_json()
    sent_json, full_json = incident_json(case.incident)
    template = Template(load_prompt(CRITIC_PROMPT_NAME, CRITIC_PROMPT_VERSION))
    fields = {
        "hypothesis_json": hypothesis,
        "remediation_kind": patch.remediation_kind.value if patch is not None else "ESCALATE",
        "diff": patch.diff if patch is not None else "",
        "regression_test": patch.regression_test if patch is not None else "",
        "dry_run_json": dry_run,
    }
    return (
        template.substitute(incident_json=sent_json, **fields),
        template.substitute(incident_json=full_json, **fields),
    )


def run_critic(
    state: MeridianState,
    *,
    model_name: str,
    chat: CriticChat,
    prompt_version: str = CRITIC_PROMPT_VERSION,
) -> dict[str, object]:
    """Review the current patch. A rejection on round 3 escalates."""
    if state.aborted:
        return {"aborted": True, "abort_reason": state.abort_reason}
    if len(state.cases) != 1:
        raise ValueError("a critic branch handles one incident")
    key, case = next(iter(state.cases.items()))
    if case.patch is None:
        raise ValueError("critic runs after a patch")
    sent, uncut = render_critic_prompt(state)
    messages: list[BaseMessage] = [
        tagged_human(sent, before=estimate_tokens(uncut), after=estimate_tokens(sent))
    ]
    if output_method_of(chat) in {"submit_tool", "json_mode"}:
        try:
            parsed, calls = run_submit_only(
                chat,
                messages,
                VerdictDraft,
                tool_name="submit_verdict",
                node=NODE,
                model_name=model_name,
                prompt_version=prompt_version,
                text_retry=CRITIC_CORRECTIVE,
            )
        except RunAborted as exc:
            return {"aborted": True, "abort_reason": str(exc)}
        draft = parsed if isinstance(parsed, VerdictDraft) else None
        over_broad = case.dry_run is not None and case.dry_run.over_broad
        verdict = _verdict(draft, case.patch.id, model_name, case.critic_rounds + 1, over_broad)
        if any(item.corrective for item in calls):
            verdict = verdict.model_copy(update={"corrective": True})
        updated = record_verdict(case, verdict)
        return {"cases": {key: updated}, "llm_calls": calls}
    structured = bind_structured(chat, VerdictDraft)
    try:
        draft, batch = _invoke(
            structured,
            [tagged_human(sent, before=estimate_tokens(uncut), after=estimate_tokens(sent))],
            model_name,
            prompt_version,
        )
    except RunAborted as exc:
        return {"aborted": True, "abort_reason": str(exc)}
    over_broad = case.dry_run is not None and case.dry_run.over_broad
    verdict = _verdict(draft, case.patch.id, model_name, case.critic_rounds + 1, over_broad)
    updated = record_verdict(case, verdict)
    return {"cases": {key: updated}, "llm_calls": batch}


def _verdict(
    draft: VerdictDraft | None,
    patch_id: object,
    model_name: str,
    round_number: int,
    over_broad: bool,
) -> Verdict:
    from uuid import UUID

    if not isinstance(patch_id, UUID):
        raise TypeError("patch id must be a UUID")
    if draft is None:
        return Verdict(
            patch_id=patch_id,
            reviewer_model=model_name,
            approved=False,
            objections=[],
            critic_degraded=True,
            round=round_number,
        )
    if not draft.approved:
        objections = _objections(draft.objections, over_broad=over_broad)
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


def _objections(raw: list[dict[str, str]] | None, *, over_broad: bool) -> list[Objection]:
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
        if kind is ObjectionKind.OVER_BROAD_CHANGE and not over_broad:
            continue
        found.append(objection(kind, message))
    return found


def _invoke(
    bound: Any,
    messages: list[BaseMessage],
    model_name: str,
    prompt_version: str,
) -> tuple[VerdictDraft | None, list[LLMCall]]:
    started = time.perf_counter()
    result, tokens_before, tokens_after = call_model(bound, messages)
    draft = _draft_from(result)
    raw = result.get("raw") if isinstance(result, dict) else result
    batch = assemble_calls(
        node=NODE,
        model=model_name,
        prompt_version=prompt_version,
        raw=raw,
        latency_ms=max(int((time.perf_counter() - started) * 1000), 0),
        attempt=1,
        succeeded=draft is not None,
        bound=bound,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        structured=True,
    )
    return draft, batch


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
