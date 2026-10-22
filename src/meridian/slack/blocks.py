"""Block Kit for the approval gate. Snapshots in the tests pin this JSON."""

import json
from dataclasses import dataclass

from meridian.contracts.approval import ApprovalDecision
from meridian.contracts.incident_case import IncidentCase, attempt_history


@dataclass(frozen=True)
class ApprovalView:
    """The fields the approval and escalation messages show."""

    severity: str
    dataset: str
    partition: str
    fault: str
    blast_radius: str
    root_cause: str
    evidence: str
    remediation_kind: str
    rows_fixed: str
    rows_changed_outside: str
    checks: str
    verdicts: str
    patch_id: str
    incident_id: str


def case_view(case: IncidentCase) -> ApprovalView:
    """Flatten one case into the message fields."""
    severity = case.severity.value if case.severity is not None else "unset"
    blast = ", ".join(case.incident.blast_radius) or "none"
    if case.hypothesis is None:
        cause = "No hypothesis was recorded."
        evidence = "none"
    else:
        cause = case.hypothesis.root_cause_summary
        refs = [item.reference for item in case.hypothesis.evidence]
        evidence = ", ".join(refs) if refs else "none"
    kind = case.patch.remediation_kind.value if case.patch is not None else "none"
    if case.dry_run is None:
        fixed = "n/a"
        outside = "n/a"
    else:
        fixed = str(case.dry_run.rows_fixed)
        outside = str(case.dry_run.rows_changed_outside)
    if case.verification is None or not case.verification.checks:
        checks = "none"
    else:
        checks = "; ".join(
            f"{item.name}={'pass' if item.passed else 'fail'} ({item.detail})"
            for item in case.verification.checks
        )
    if not case.verdicts:
        verdicts = "none"
    else:
        verdicts = "; ".join(
            f"round {item.round} {'approved' if item.approved else 'rejected'} "
            f"by {item.reviewer_model}"
            for item in case.verdicts
        )
    return ApprovalView(
        severity=severity,
        dataset=case.incident.dataset,
        partition=case.incident.partition,
        fault=case.incident.fault_type.value,
        blast_radius=blast,
        root_cause=cause,
        evidence=evidence,
        remediation_kind=kind,
        rows_fixed=fixed,
        rows_changed_outside=outside,
        checks=checks,
        verdicts=verdicts,
        patch_id="" if case.patch is None else str(case.patch.id),
        incident_id=str(case.incident.id),
    )


def approval_blocks(view: ApprovalView, thread_id: str) -> list[dict[str, object]]:
    """Approve and Reject. Reject opens a modal; it does not decide by itself."""
    blocks = _context(view, "Approval")
    blocks.append(
        {
            "type": "actions",
            "block_id": "meridian_actions",
            "elements": [
                _button("Approve", "meridian_approve", "primary", view, thread_id, "approve"),
                _button("Reject", "meridian_reject", "danger", view, thread_id, "reject"),
            ],
        }
    )
    return blocks


def escalation_blocks(view: ApprovalView, thread_id: str) -> list[dict[str, object]]:
    """Acknowledge only. There is no approve path."""
    blocks = _context(view, "Escalation")
    blocks.append(
        {
            "type": "actions",
            "block_id": "meridian_actions",
            "elements": [
                _button(
                    "Acknowledge",
                    "meridian_acknowledge",
                    None,
                    view,
                    thread_id,
                    "acknowledge",
                )
            ],
        }
    )
    return blocks


def decided_blocks(view: ApprovalView, decision: ApprovalDecision) -> list[dict[str, object]]:
    """The same context with the decision in place of the buttons."""
    blocks = _context(view, "Decision")
    comment = f" comment={decision.comment}" if decision.comment else ""
    blocks.append(
        _section(
            f"decision={decision.decision.value} approver={decision.approver} "
            f"at={decision.decided_at.isoformat()}{comment}"
        )
    )
    return blocks


def reject_modal(
    *,
    patch_id: str,
    thread_id: str,
    channel: str,
    message_ts: str,
) -> dict[str, object]:
    """A modal whose comment field is required."""
    metadata = json.dumps(
        {
            "patch_id": patch_id,
            "thread_id": thread_id,
            "channel": channel,
            "message_ts": message_ts,
        }
    )
    return {
        "type": "modal",
        "callback_id": "meridian_reject_submit",
        "title": {"type": "plain_text", "text": "Reject patch"},
        "submit": {"type": "plain_text", "text": "Reject"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "private_metadata": metadata,
        "blocks": [
            {
                "type": "input",
                "block_id": "comment",
                "label": {"type": "plain_text", "text": "Comment"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "comment",
                    "multiline": True,
                },
            }
        ],
    }


def button_kinds(blocks: list[dict[str, object]]) -> list[str]:
    """Action kinds on the message, used by the escalation test."""
    found: list[str] = []
    for block in blocks:
        elements = block.get("elements")
        if not isinstance(elements, list):
            continue
        for element in elements:
            if not isinstance(element, dict):
                continue
            value = element.get("value")
            if not isinstance(value, str):
                continue
            loaded = json.loads(value)
            if isinstance(loaded, dict) and isinstance(loaded.get("kind"), str):
                found.append(loaded["kind"])
    return found


def _context(view: ApprovalView, title: str) -> list[dict[str, object]]:
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{title} {view.severity}"},
        },
        _section(
            f"dataset={view.dataset} partition={view.partition} "
            f"fault={view.fault} blast={view.blast_radius}"
        ),
        _section(f"root_cause={view.root_cause} evidence={view.evidence}"),
        _section(
            f"remediation_kind={view.remediation_kind} "
            f"rows_fixed={view.rows_fixed} rows_changed_outside={view.rows_changed_outside}"
        ),
        _section(f"checks={view.checks}"),
        _section(f"verdicts={view.verdicts}"),
    ]


def _section(text: str) -> dict[str, object]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": _escape(text)}}


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _button(
    label: str,
    action_id: str,
    style: str | None,
    view: ApprovalView,
    thread_id: str,
    kind: str,
) -> dict[str, object]:
    button: dict[str, object] = {
        "type": "button",
        "text": {"type": "plain_text", "text": label},
        "action_id": action_id,
        "value": json.dumps(
            {
                "thread_id": thread_id,
                "patch_id": view.patch_id,
                "incident_id": view.incident_id,
                "kind": kind,
            }
        ),
    }
    if style is not None:
        button["style"] = style
    return button


def attempt_count(case: IncidentCase) -> int:
    """Closed attempts plus the open patch. Used by the postmortem timeline."""
    return len(attempt_history(case))
