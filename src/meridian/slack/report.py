"""Deterministic postmortem. No model call."""

from datetime import timedelta
from decimal import Decimal

from meridian.contracts.incident_case import IncidentCase, attempt_history
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.postmortem import Postmortem, TimelineEvent
from meridian.llm.prices import cost_usd, load_prices

_PAID = frozenset({"anthropic", "xai"})


def ledger_usd(calls: list[LLMCall]) -> float:
    """USD from the price table. Free providers, and unpaid rows, are zero."""
    prices = load_prices()
    total = Decimal(0)
    for call in calls:
        if call.provider not in _PAID:
            continue
        price = prices.get(call.model)
        if price is None:
            continue
        total += cost_usd(price, call.input_tokens, call.output_tokens)
    return float(total)


def build_postmortem(case: IncidentCase, calls: list[LLMCall]) -> Postmortem:
    """The contract record. Tokens stay in the markdown."""
    cause = (
        case.hypothesis.root_cause_summary
        if case.hypothesis is not None
        else "No hypothesis was recorded."
    )
    if case.approval is not None and case.patch is not None:
        summary = f"{case.approval.decision.value} {case.patch.remediation_kind.value}"
        if case.approval.comment:
            summary = f"{summary}: {case.approval.comment}"
    elif case.patch is not None:
        summary = case.patch.remediation_kind.value
    else:
        summary = "No patch was produced."
    if case.approval is None:
        elapsed = 0
    else:
        delta = case.approval.decided_at - case.incident.detected_at
        elapsed = max(int(delta.total_seconds()), 0)
    return Postmortem(
        incident_id=case.incident.id,
        timeline=build_timeline(case),
        root_cause=cause,
        fix_summary=summary,
        time_to_resolution_s=elapsed,
        cost_usd=ledger_usd(calls),
    )


def build_timeline(case: IncidentCase) -> list[TimelineEvent]:
    """detected, triaged, hypothesis, each attempt, then the decision."""
    base = case.incident.detected_at
    rows: list[tuple[str, str]] = [
        ("sentinel", f"detected {case.incident.fault_type.value} on {case.incident.dataset}"),
    ]
    if case.triage is not None:
        rows.append(("triage", f"severity {case.triage.final_severity.value}"))
    if case.hypothesis is not None:
        rows.append(("root_cause", case.hypothesis.root_cause_summary))
    for index, attempt in enumerate(attempt_history(case), start=1):
        rows.append(("fix", f"attempt {index} {attempt.patch.remediation_kind.value}"))
    if case.approval is not None:
        text = f"{case.approval.decision.value} by {case.approval.approver}"
        if case.approval.comment:
            text = f"{text}: {case.approval.comment}"
        rows.append(("human", text))
    events: list[TimelineEvent] = []
    for offset, (actor, summary) in enumerate(rows):
        moment = base + timedelta(seconds=offset)
        if actor == "human" and case.approval is not None and case.approval.decided_at >= moment:
            moment = case.approval.decided_at
        events.append(TimelineEvent(occurred_at=moment, actor=actor, summary=summary))
    return events


def render_markdown(case: IncidentCase, report: Postmortem, calls: list[LLMCall]) -> str:
    """The file posted to the thread and written under docs/postmortems."""
    incoming = sum(call.input_tokens for call in calls)
    outgoing = sum(call.output_tokens for call in calls)
    lines = [
        f"# Postmortem {case.incident.id}",
        "",
        f"- Root cause: {report.root_cause}",
        f"- Fix: {report.fix_summary}",
        f"- Time to resolution (s): {report.time_to_resolution_s}",
        f"- Tokens: input {incoming} output {outgoing}",
        f"- USD: {report.cost_usd:.2f} (budget ledger)",
        "",
        "## Timeline",
        "",
    ]
    for event in report.timeline:
        lines.append(f"- {event.occurred_at.isoformat()} {event.actor}: {event.summary}")
    lines.append("")
    return "\n".join(lines)
