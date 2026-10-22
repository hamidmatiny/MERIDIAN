"""One incident and the records produced while responding to it.

Loop limits from AGENTS.md live on the case, not on the run:

* Fix and Critic may exchange at most 3 rounds (``critic_rounds`` <= 3).
* A failed verification may route back to Fix at most 2 times. ``fix_attempts``
  counts Fix executions, so the cap is 3: the original attempt plus two retries.
"""

from typing import Final, Self
from uuid import UUID

from pydantic import Field, model_validator

from meridian.contracts.approval import ApprovalDecision
from meridian.contracts.base import MeridianModel
from meridian.contracts.dry_run import DryRunReport
from meridian.contracts.hypothesis import Hypothesis
from meridian.contracts.incident import Incident
from meridian.contracts.patch import Patch
from meridian.contracts.postmortem import Postmortem
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.severity import Severity
from meridian.contracts.triage import TriageResult
from meridian.contracts.verdict import Verdict
from meridian.contracts.verification import VerificationResult

# Fix↔Critic max 3 rounds.
MAX_CRITIC_ROUNDS: Final = 3
# One initial Fix execution plus two verifier-driven reroutes.
MAX_FIX_ATTEMPTS: Final = 3


class Attempt(MeridianModel):
    """One Fix execution and everything recorded for it.

    A later attempt appends another ``Attempt``. It does not replace this one.
    """

    patch: Patch
    dry_run: DryRunReport | None = Field(
        default=None,
        description="Dry-run for this attempt. None when the attempt did not reach one.",
    )
    verdicts: list[Verdict] = Field(
        default_factory=list,
        description="Critic verdicts recorded for this attempt.",
    )
    verification: VerificationResult | None = Field(
        default=None,
        description="Verifier result for this attempt. None until the verifier runs.",
    )

    @model_validator(mode="after")
    def ids_match(self) -> Self:
        """Keep this attempt's reviews pointed at its own patch."""
        for verdict in self.verdicts:
            _same(verdict.patch_id, self.patch.id, "attempt verdict.patch_id must equal patch.id")
        if self.verification is not None:
            _same(
                self.verification.patch_id,
                self.patch.id,
                "attempt verification.patch_id must equal patch.id",
            )
        return self


def _same(child_id: UUID, parent_id: UUID, message: str) -> None:
    if child_id != parent_id:
        raise ValueError(message)


class IncidentCase(MeridianModel):
    """The response record for a single incident.

    Effective severity is ``triage.final_severity``. ``Incident.severity`` stays unset.
    """

    incident: Incident = Field(description="The detected incident. Severity on it stays unset.")
    triage: TriageResult | None = Field(
        default=None,
        description="Triage decision for this incident, once Triage has run.",
    )
    hypothesis: Hypothesis | None = Field(
        default=None,
        description="Current root-cause hypothesis, if one has been proposed.",
    )
    patch: Patch | None = Field(
        default=None,
        description="Current patch, if Fix has produced one.",
    )
    verdicts: list[Verdict] = Field(
        default_factory=list,
        description="Critic verdicts for this case, in the order they were recorded.",
    )
    verification: VerificationResult | None = Field(
        default=None,
        description="Latest deterministic replay result, if any.",
    )
    approval: ApprovalDecision | None = Field(
        default=None,
        description="Human approval decision, if the gate has completed.",
    )
    postmortem: Postmortem | None = Field(
        default=None,
        description="Incident report, if the loop has finished.",
    )
    fix_attempts: int = Field(
        default=0,
        ge=0,
        le=MAX_FIX_ATTEMPTS,
        description="Fix executions so far. Maximum is 3 (initial attempt plus two retries).",
    )
    critic_rounds: int = Field(
        default=0,
        ge=0,
        le=MAX_CRITIC_ROUNDS,
        description="Completed Fix/Critic rounds so far. Maximum is 3.",
    )
    needs_human: bool = Field(
        default=False,
        description="True when the case escalates or the loop cap is hit.",
    )
    critic_rejection_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of critic verdicts on this case that rejected the patch.",
    )
    pre_critic_attempt: int = Field(
        default=0,
        ge=0,
        description="Fix attempt last checked. A blocked check is not a critic round.",
    )
    pre_critic_blocked: bool = Field(
        default=False,
        description="True when syntax or the sandbox gate sent the patch back to Fix.",
    )
    pre_critic_note: str = Field(
        default="",
        description="Why the pre-critic bounced the patch. Empty when it passed.",
    )
    dry_run: DryRunReport | None = Field(
        default=None,
        description="Latest pre-critic dry-run. The critic is shown this report.",
    )
    hypothesis_from_budget_fallback: bool = Field(
        default=False,
        description="True when the hypothesis is the tool-budget fallback.",
    )
    attempts: list[Attempt] = Field(
        default_factory=list,
        description="Closed Fix attempts. The open attempt stays on the case fields.",
    )

    @property
    def severity(self) -> Severity | None:
        """Severity from triage. None until this case has a triage result."""
        if self.triage is None:
            return None
        return self.triage.final_severity

    @model_validator(mode="after")
    def parent_ids_match(self) -> Self:
        """Reject a case whose child records point at a different parent."""
        if self.triage is not None:
            _same(
                self.triage.incident_id,
                self.incident.id,
                "triage.incident_id must equal incident.id",
            )
        if self.hypothesis is not None:
            _same(
                self.hypothesis.incident_id,
                self.incident.id,
                "hypothesis.incident_id must equal incident.id",
            )
        if self.patch is not None:
            _same(
                self.patch.incident_id,
                self.incident.id,
                "patch.incident_id must equal incident.id",
            )
        if self.hypothesis is not None and self.patch is not None:
            _same(
                self.patch.hypothesis_id,
                self.hypothesis.id,
                "patch.hypothesis_id must equal hypothesis.id",
            )
        if self.patch is not None:
            for verdict in self.verdicts:
                _same(
                    verdict.patch_id,
                    self.patch.id,
                    "verdict.patch_id must equal patch.id",
                )
            if self.verification is not None:
                _same(
                    self.verification.patch_id,
                    self.patch.id,
                    "verification.patch_id must equal patch.id",
                )
            if self.approval is not None:
                _same(
                    self.approval.patch_id,
                    self.patch.id,
                    "approval.patch_id must equal patch.id",
                )
        if self.postmortem is not None:
            _same(
                self.postmortem.incident_id,
                self.incident.id,
                "postmortem.incident_id must equal incident.id",
            )
        if (
            self.patch is not None
            and self.patch.remediation_kind is RemediationKind.ESCALATE
            and not self.needs_human
        ):
            raise ValueError("ESCALATE requires needs_human")
        return self


def attempt_history(case: IncidentCase) -> list[Attempt]:
    """Closed attempts plus the open one. A stored attempt is not rewritten."""
    closed = list(case.attempts)
    if case.patch is None:
        return closed
    used = sum(len(item.verdicts) for item in closed)
    current = Attempt(
        patch=case.patch,
        dry_run=case.dry_run,
        verdicts=list(case.verdicts[used:]),
        verification=case.verification,
    )
    if closed and closed[-1] == current:
        return closed
    return [*closed, current]
