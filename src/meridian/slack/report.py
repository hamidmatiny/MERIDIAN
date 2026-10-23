"""Deterministic postmortem. No model call.

Node times come from recorded progress events. A run that has none is marked
``times not recorded (pre-5.2 run)`` instead of being given synthetic offsets.
"""

from datetime import datetime
from decimal import Decimal

from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.postmortem import Postmortem, TimelineEvent
from meridian.contracts.progress_event import ProgressEvent
from meridian.llm.prices import cost_usd, load_prices
from meridian.slack.text import mention_user

_PAID = frozenset({"anthropic", "xai"})
TIMES_NOT_RECORDED = "times not recorded (pre-5.2 run)"


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


def build_postmortem(
    case: IncidentCase,
    calls: list[LLMCall],
    *,
    events: list[ProgressEvent] | None = None,
    posted_at: datetime | None = None,
) -> Postmortem:
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
    matched = _for_case(case, [] if events is None else events)
    machine, human = _waits(case, matched, posted_at)
    elapsed = 0 if machine is None or human is None else machine + human
    return Postmortem(
        incident_id=case.incident.id,
        timeline=build_timeline(case, matched),
        root_cause=cause,
        fix_summary=summary,
        time_to_resolution_s=elapsed,
        machine_time_s=machine,
        human_wait_s=human,
        cost_usd=ledger_usd(calls),
    )


def build_timeline(case: IncidentCase, events: list[ProgressEvent]) -> list[TimelineEvent]:
    """One row per recorded node boundary, then the decision at its real time."""
    if not events:
        rows = [
            TimelineEvent(
                occurred_at=case.incident.detected_at,
                actor="system",
                summary=TIMES_NOT_RECORDED,
            )
        ]
        decision = _decision_event(case)
        if decision is not None:
            rows.append(decision)
        return sorted(rows, key=lambda item: item.occurred_at)
    rows = [
        TimelineEvent(
            occurred_at=event.occurred_at,
            actor=event.node,
            summary=f"attempt {event.attempt} {event.phase}",
        )
        for event in events
    ]
    decision = _decision_event(case)
    if decision is not None:
        rows.append(decision)
    return sorted(rows, key=lambda item: item.occurred_at)


def render_markdown(case: IncidentCase, report: Postmortem, calls: list[LLMCall]) -> str:
    """The file posted to the thread and written under docs/postmortems."""
    incoming = sum(call.input_tokens for call in calls)
    outgoing = sum(call.output_tokens for call in calls)
    recorded = report.machine_time_s is not None
    lines = [
        f"# Postmortem {case.incident.id}",
        "",
        f"- Root cause: {report.root_cause}",
        f"- Fix: {report.fix_summary}",
        f"- Machine time (s): {_shown(report.machine_time_s)}",
        f"- Human wait (s): {_shown(report.human_wait_s)}",
        f"- Time to resolution (s): {_shown(report.time_to_resolution_s if recorded else None)}",
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


def _for_case(case: IncidentCase, events: list[ProgressEvent]) -> list[ProgressEvent]:
    incident_id = str(case.incident.id)
    return [event for event in events if event.case_id in {incident_id, "-"}]


def _waits(
    case: IncidentCase,
    events: list[ProgressEvent],
    posted_at: datetime | None,
) -> tuple[int | None, int | None]:
    """Detection to slack_approval, and post to decision. Both, or neither."""
    reached = [
        event.occurred_at
        for event in events
        if event.node == "slack_approval" and event.phase == "start"
    ]
    if not reached or posted_at is None or case.approval is None:
        return None, None
    machine = max(int((min(reached) - case.incident.detected_at).total_seconds()), 0)
    human = max(int((case.approval.decided_at - posted_at).total_seconds()), 0)
    return machine, human


def _decision_event(case: IncidentCase) -> TimelineEvent | None:
    if case.approval is None:
        return None
    text = f"{case.approval.decision.value} by {mention_user(case.approval.approver)}"
    if case.approval.comment:
        text = f"{text}: {case.approval.comment}"
    return TimelineEvent(occurred_at=case.approval.decided_at, actor="human", summary=text)


def _shown(value: int | None) -> str:
    return "not recorded" if value is None else str(value)
