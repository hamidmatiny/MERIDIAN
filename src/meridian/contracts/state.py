"""LangGraph state for the incident-response graph.

The graph itself starts at Sentinel and stops after Triage. Later nodes are planned.
This model is the only state those nodes will share.

``partitions`` is the Sentinel input. ``incidents`` is the batch Sentinel emits.
``incident`` remains the single record a later phase attaches a hypothesis to.

Loop limits from AGENTS.md:

* Fix and Critic may exchange at most 3 rounds (``critic_rounds`` <= 3).
* A failed verification may route back to Fix at most 2 times. ``fix_attempts``
  counts Fix executions, so the cap is 3: the original attempt plus two retries.
"""

from typing import Final, Self
from uuid import UUID

from pydantic import Field, model_validator

from meridian.contracts.approval import ApprovalDecision
from meridian.contracts.base import MeridianModel
from meridian.contracts.hypothesis import Hypothesis
from meridian.contracts.incident import Incident
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.patch import Patch
from meridian.contracts.postmortem import Postmortem
from meridian.contracts.triage import TriageResult
from meridian.contracts.verdict import Verdict
from meridian.contracts.verification import VerificationResult

# Fix↔Critic max 3 rounds.
MAX_CRITIC_ROUNDS: Final = 3
# One initial Fix execution plus two verifier-driven reroutes.
MAX_FIX_ATTEMPTS: Final = 3


def _same(child_id: UUID, parent_id: UUID, message: str) -> None:
    if child_id != parent_id:
        raise ValueError(message)


class MeridianState(MeridianModel):
    """Partial graph state. Later nodes fill fields; parent ids must agree when both exist."""

    incident: Incident | None = Field(
        default=None,
        description="Incident under response, if detected.",
    )
    hypothesis: Hypothesis | None = Field(
        default=None,
        description="Current root-cause hypothesis, if one has been proposed.",
    )
    patch: Patch | None = Field(
        default=None,
        description="Current patch, if Fix has produced one.",
    )
    verdict: Verdict | None = Field(
        default=None,
        description="Latest critic verdict, if any.",
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
    partitions: list[PartitionRef] = Field(
        default_factory=list,
        description="Partitions Sentinel evaluates.",
    )
    incidents: list[Incident] = Field(
        default_factory=list,
        description="Incidents Sentinel emitted, deduplicated by id.",
    )
    triage_results: list[TriageResult] = Field(
        default_factory=list,
        description="One triage decision per incident.",
    )
    llm_calls: list[LLMCall] = Field(
        default_factory=list,
        description="Every model attempt made during the run.",
    )

    @model_validator(mode="after")
    def parent_ids_match(self) -> Self:
        """Reject a state whose child records point at a different parent."""
        if self.incident is not None and self.hypothesis is not None:
            _same(
                self.hypothesis.incident_id,
                self.incident.id,
                "hypothesis.incident_id must equal incident.id",
            )
        if self.incident is not None and self.patch is not None:
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
        if self.patch is not None and self.verdict is not None:
            _same(
                self.verdict.patch_id,
                self.patch.id,
                "verdict.patch_id must equal patch.id",
            )
        if self.patch is not None and self.verification is not None:
            _same(
                self.verification.patch_id,
                self.patch.id,
                "verification.patch_id must equal patch.id",
            )
        if self.patch is not None and self.approval is not None:
            _same(
                self.approval.patch_id,
                self.patch.id,
                "approval.patch_id must equal patch.id",
            )
        if self.incident is not None and self.postmortem is not None:
            _same(
                self.postmortem.incident_id,
                self.incident.id,
                "postmortem.incident_id must equal incident.id",
            )
        return self
