"""Turn one finished graph state into the JSON a report can aggregate."""

from typing import Any

import pandas as pd

from meridian.bench.score_fix import score as score_fix
from meridian.bench.score_root_cause import score as score_root_cause
from meridian.contracts.injected_fault import InjectedFault
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.progress import load_progress
from meridian.slack.report import ledger_usd


def summarize_run(
    state: MeridianState,
    *,
    split: str,
    dataset: str,
    seed: int,
    fault: str | None,
    intensity: float | None,
    clean: pd.DataFrame,
    label: InjectedFault | None,
    wall_s: float,
    wall_abort: bool,
    run_key: str,
) -> dict[str, Any]:
    """One completed run. The approval decision is BENCH_NO_HUMAN and nothing was applied."""
    incidents = [
        {"id": str(case.incident.id), "fault": case.incident.fault_type.value}
        for case in state.cases.values()
    ]
    cases = [_case_row(case, clean, label) for case in state.cases.values()]
    calls = list(state.llm_calls)
    return {
        "run_key": run_key,
        "split": split,
        "dataset": dataset,
        "seed": seed,
        "fault": fault,
        "intensity": intensity,
        "decision": "BENCH_NO_HUMAN",
        "applied": False,
        "completed": True,
        "wall_abort": wall_abort or _wall_abort(state),
        "abort_reason": state.abort_reason,
        "wall_s": wall_s,
        "machine_time_s": _machine_time(state),
        "incidents": incidents,
        "cases": cases,
        "input_tokens": sum(call.input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "usd": ledger_usd(calls),
        "timeouts": sum(1 for call in calls if call.timed_out),
        "structured": _structured(calls),
        "node_latency_ms": _latency(calls),
    }


def _case_row(case: object, clean: pd.DataFrame, label: InjectedFault | None) -> dict[str, Any]:
    from meridian.contracts.incident_case import IncidentCase

    if not isinstance(case, IncidentCase):
        raise TypeError("case must be IncidentCase")
    triage = case.triage
    hypothesis = case.hypothesis
    patch = case.patch
    verification = case.verification
    fix_score = None
    root_score = None
    if label is not None and hypothesis is not None:
        root_score = score_root_cause(hypothesis, label).model_dump()
    if label is not None and patch is not None:
        fix_score = score_fix(case, clean, label).model_dump()
    last = case.verdicts[-1] if case.verdicts else None
    floor_raised = False
    if triage is not None and triage.llm_severity is not None:
        floor_raised = triage.final_severity is not triage.llm_severity
    return {
        "fault": case.incident.fault_type.value,
        "severity": None if triage is None else triage.final_severity.value,
        "llm_severity": None
        if triage is None or triage.llm_severity is None
        else triage.llm_severity.value,
        "floor_severity": None if triage is None else triage.floor_severity.value,
        "floor_raised": floor_raised,
        "raw_causal_correct": None if root_score is None else root_score["raw_causal_correct"],
        "causal_event_correct": None if root_score is None else root_score["causal_event_correct"],
        "columns_jaccard": None if root_score is None else root_score["columns_jaccard"],
        "grounding_violations": 0 if hypothesis is None else hypothesis.grounding_violations,
        "guardrail_fn": bool(
            root_score is not None
            and root_score["raw_causal_correct"]
            and not root_score["causal_event_correct"]
        ),
        "kind": None if patch is None else patch.remediation_kind.value,
        "kind_appropriate": None if fix_score is None else fix_score["kind_appropriate"],
        "repaired_matches_clean": None
        if fix_score is None
        else fix_score["repaired_matches_clean"],
        "verified": bool(verification is not None and verification.passed),
        "escalated": bool(patch is not None and patch.remediation_kind is RemediationKind.ESCALATE),
        "critic_approved": None if last is None else last.approved,
        "critic_degraded": any(item.critic_degraded for item in case.verdicts),
        "critic_corrective": any(item.corrective for item in case.verdicts),
        "has_hypothesis": hypothesis is not None,
        "has_verdict": last is not None,
    }


def _wall_abort(state: MeridianState) -> bool:
    """True when the graph stopped because the run clock ran out, whatever the wording."""
    return bool(state.aborted and "WALL_BUDGET" in state.abort_reason)


def _machine_time(state: MeridianState) -> float | None:
    events = load_progress(state.run_id)
    reached = [
        event.occurred_at
        for event in events
        if event.node == "slack_approval" and event.phase == "start"
    ]
    if not reached or not state.cases:
        return None
    detected = min(case.incident.detected_at for case in state.cases.values())
    return max((min(reached) - detected).total_seconds(), 0.0)


def _structured(calls: list[Any]) -> dict[str, dict[str, int]]:
    found: dict[str, dict[str, int]] = {}
    for call in calls:
        if not call.structured:
            continue
        slot = found.setdefault(call.node, {"ok": 0, "n": 0})
        slot["n"] += 1
        if call.succeeded:
            slot["ok"] += 1
    return found


def _latency(calls: list[Any]) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for call in calls:
        found.setdefault(call.node, []).append(int(call.latency_ms))
    return found
