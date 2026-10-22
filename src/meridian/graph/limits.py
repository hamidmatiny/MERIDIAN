"""Fix / critic / verifier routing. The counters cannot move past the case caps."""

from uuid import UUID

from meridian.contracts.incident_case import (
    MAX_CRITIC_ROUNDS,
    MAX_FIX_ATTEMPTS,
    Attempt,
    IncidentCase,
)
from meridian.contracts.patch import Patch
from meridian.contracts.remediation import Objection, ObjectionKind, RemediationKind
from meridian.contracts.verdict import Verdict
from meridian.contracts.verification import VerificationResult


def route_after_root(case: IncidentCase) -> str:
    """A hypothesis goes to Fix. Anything else ends the branch."""
    if case.hypothesis is None or case.needs_human:
        return "end"
    return "fix"


def route_after_critic(case: IncidentCase) -> str:
    """Approve goes to the verifier. A degraded review is not a rejection."""
    if not case.verdicts:
        return "end"
    latest = case.verdicts[-1]
    escalate = case.patch is not None and case.patch.remediation_kind is RemediationKind.ESCALATE
    # A rejected escalation stops. An approved one still reaches the verifier.
    if case.needs_human and not (escalate and latest.approved and case.verification is None):
        return "end"
    if latest.approved:
        return "verifier"
    if latest.critic_degraded:
        return "critic"
    if case.critic_rounds >= MAX_CRITIC_ROUNDS or case.fix_attempts >= MAX_FIX_ATTEMPTS:
        return "end"
    return "fix"


def route_after_verifier(case: IncidentCase) -> str:
    """A pass or an escalation ends the branch. A failure under the cap returns to Fix."""
    if case.needs_human:
        return "end"
    if case.verification is not None and case.verification.passed:
        return "end"
    if case.fix_attempts >= MAX_FIX_ATTEMPTS:
        return "end"
    return "fix"


def begin_fix(case: IncidentCase) -> IncidentCase:
    """Count one Fix execution. Archive the previous attempt before clearing it."""
    if case.fix_attempts >= MAX_FIX_ATTEMPTS:
        raise ValueError("fix attempt cap")
    attempts = list(case.attempts)
    if case.patch is not None:
        used = sum(len(item.verdicts) for item in attempts)
        attempts.append(
            Attempt(
                patch=case.patch,
                dry_run=case.dry_run,
                verdicts=list(case.verdicts[used:]),
                verification=case.verification,
            )
        )
    return case.model_copy(
        update={
            "fix_attempts": case.fix_attempts + 1,
            "verification": None,
            "dry_run": None,
            "pre_critic_blocked": False,
            "pre_critic_note": "",
            "attempts": attempts,
        },
    )


def finish_fix(case: IncidentCase, patch: Patch) -> IncidentCase:
    """Store the patch produced by this Fix execution.

    An ESCALATE patch always needs a human. The case validator rejects the
    other combination, so the flag is set in the same update as the patch.
    """
    needs = case.needs_human or patch.remediation_kind is RemediationKind.ESCALATE
    return case.model_copy(update={"patch": patch, "needs_human": needs})


def record_verdict(case: IncidentCase, verdict: Verdict) -> IncidentCase:
    """Append a verdict. Round 3 of rejection, or a rejection with no Fix budget, escalates."""
    if case.critic_rounds >= MAX_CRITIC_ROUNDS:
        raise ValueError("critic round cap")
    rounds = case.critic_rounds + 1
    if verdict.round != rounds:
        raise ValueError("verdict.round must equal the next critic round")
    verdicts = [*case.verdicts, verdict]
    needs = case.needs_human
    if case.patch is not None and case.patch.remediation_kind is RemediationKind.ESCALATE:
        needs = True
    degraded = sum(1 for item in verdicts if item.critic_degraded)
    if verdict.critic_degraded:
        if degraded >= 2:
            needs = True
    elif not verdict.approved and (
        rounds >= MAX_CRITIC_ROUNDS or case.fix_attempts >= MAX_FIX_ATTEMPTS
    ):
        needs = True
    return case.model_copy(
        update={
            "verdicts": verdicts,
            "critic_rounds": rounds,
            "critic_rejection_rate": rejection_rate(verdicts),
            "needs_human": needs,
        },
    )


def record_verification(
    case: IncidentCase,
    result: VerificationResult,
    *,
    escalate: bool,
) -> IncidentCase:
    """Store the replay. Escalation and a failure at the Fix cap both need a human."""
    needs = case.needs_human or escalate
    if not result.passed and case.fix_attempts >= MAX_FIX_ATTEMPTS:
        needs = True
    return case.model_copy(update={"verification": result, "needs_human": needs})


def rejection_rate(verdicts: list[Verdict]) -> float:
    """Share of verdicts that rejected. Zero when the critic has not run."""
    if not verdicts:
        return 0.0
    rejected = sum(1 for item in verdicts if not item.approved and not item.critic_degraded)
    return rejected / len(verdicts)


def objection(kind: ObjectionKind, message: str) -> Objection:
    """Build one objection. Callers pass a non-blank message."""
    return Objection(kind=kind, message=message)


def patch_id_for(incident_id: UUID, prompt_version: str) -> UUID:
    """One patch id per incident and prompt, so later verdicts still match."""
    from uuid import uuid5

    from meridian.contracts.namespace import MERIDIAN_NAMESPACE

    return uuid5(MERIDIAN_NAMESPACE, f"patch|{incident_id}|{prompt_version}")


def escalate_source() -> str:
    """Module body for a patch that does not transform rows."""
    return "def repair(df):\n    return df\n"


def kind_or_escalate(value: str) -> RemediationKind:
    """Map a model string onto a kind. An unknown value escalates."""
    try:
        return RemediationKind(value)
    except ValueError:
        return RemediationKind.ESCALATE
