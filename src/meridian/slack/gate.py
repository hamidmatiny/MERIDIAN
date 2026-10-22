"""Slack approval node and the postmortem that follows it.

``interrupt`` returns on resume. Code before it runs again, so the post is keyed.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langgraph.types import interrupt

from meridian.contracts.approval import ApprovalDecision, ApprovalOutcome
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.state import MeridianState
from meridian.slack.apply import apply_patch
from meridian.slack.blocks import (
    approval_blocks,
    case_view,
    decided_blocks,
    escalation_blocks,
)
from meridian.slack.gateway import SlackGateway
from meridian.slack.report import build_postmortem, render_markdown
from meridian.slack.store import OpenCase, OpenIndex


@dataclass(frozen=True)
class ApprovalPorts:
    """Directories and the Slack client the gate closes over."""

    gateway: SlackGateway
    apply_root: Path
    postmortem_root: Path
    channel: str
    open_index: OpenIndex


def run_approval(state: MeridianState, ports: ApprovalPorts) -> dict[str, object]:
    """Post the request, then stop until a person or the timeout resumes the graph."""
    if len(state.cases) != 1:
        raise ValueError("approval handles one incident")
    key, case = next(iter(state.cases.items()))
    view = case_view(case)
    escalation = _needs_escalation(case)
    thread_id = str(state.run_id)
    blocks = escalation_blocks(view, thread_id) if escalation else approval_blocks(view, thread_id)
    title = "Escalation" if escalation else "Approval"
    message_key = view.patch_id or view.incident_id
    ts = ports.gateway.post_once(
        key=message_key,
        channel=ports.channel,
        text=title,
        blocks=blocks,
    )
    if case.patch is not None:
        ports.gateway.upload_once(
            key=message_key,
            channel=ports.channel,
            thread_ts=ts,
            filename=f"{case.incident.id}.diff",
            content=case.patch.diff,
        )
    ports.open_index.upsert(_open_case(case, thread_id))
    raw = interrupt({"patch_id": message_key, "thread_id": thread_id})
    if not isinstance(raw, dict):
        raise TypeError("approval resume must be an object")
    decision = _decision(case, raw)
    needs = case.needs_human
    if decision is None or decision.decision is not ApprovalOutcome.APPROVED:
        needs = True
    updated = (
        case
        if decision is None
        else case.model_copy(update={"approval": decision, "needs_human": needs})
    )
    if decision is not None and decision.decision is ApprovalOutcome.APPROVED and _can_apply(case):
        apply_patch(updated, ports.apply_root)
    if decision is not None:
        ports.gateway.show_decision(
            key=message_key,
            channel=ports.channel,
            text=title,
            blocks=decided_blocks(view, decision),
        )
    return {"cases": {key: updated}}


def run_postmortem(state: MeridianState, ports: ApprovalPorts) -> dict[str, object]:
    """Write the report and post it on the approval thread. Every finished case."""
    if len(state.cases) != 1:
        raise ValueError("postmortem handles one incident")
    key, case = next(iter(state.cases.items()))
    if case.postmortem is not None:
        return {"cases": {key: case}}
    report = build_postmortem(case, list(state.llm_calls))
    folder = ports.postmortem_root
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{case.incident.id}.md"
    path.write_text(render_markdown(case, report, list(state.llm_calls)), encoding="utf-8")
    message_key = str(case.patch.id) if case.patch is not None else str(case.incident.id)
    ports.gateway.post_once(
        key=f"{message_key}:postmortem",
        channel=ports.channel,
        text=path.read_text(encoding="utf-8"),
        blocks=[],
        thread_ts=ports.gateway.thread_ts(message_key),
    )
    ports.open_index.remove(str(case.incident.id))
    return {"cases": {key: case.model_copy(update={"postmortem": report})}}


def _needs_escalation(case: IncidentCase) -> bool:
    if case.needs_human:
        return True
    return case.verification is None or not case.verification.passed


def _can_apply(case: IncidentCase) -> bool:
    return (
        not case.needs_human
        and case.verification is not None
        and case.verification.passed
        and case.patch is not None
    )


def _decision(case: IncidentCase, raw: dict[str, Any]) -> ApprovalDecision | None:
    if case.patch is None:
        return None
    comment = raw.get("comment", "")
    return ApprovalDecision(
        patch_id=UUID(str(raw.get("patch_id", case.patch.id))),
        decision=ApprovalOutcome(str(raw["decision"])),
        approver=str(raw["approver"]),
        decided_at=_moment(raw["decided_at"]),
        comment="" if comment is None else str(comment),
    )


def _moment(value: object) -> datetime:
    """The checkpoint stores the resume payload as strings."""
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _open_case(case: IncidentCase, thread_id: str) -> OpenCase:
    severity = case.severity.value if case.severity is not None else "unset"
    return OpenCase(
        incident_id=str(case.incident.id),
        severity=severity,
        node="slack_approval",
        needs_human=case.needs_human or _needs_escalation(case),
        detected_at=case.incident.detected_at,
        posted_at=datetime.now(UTC),
        thread_id=thread_id,
        patch_id="" if case.patch is None else str(case.patch.id),
    )
