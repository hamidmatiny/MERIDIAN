"""Remediation kinds and the critic's objection taxonomy."""

from enum import StrEnum

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank


class RemediationKind(StrEnum):
    """How a patch tries to clear an incident.

    Not every fault has a transform. Nulls stay null, and a late partition needs
    a rerun. The fix step is allowed to refuse a code repair.
    """

    DATA_REPAIR = "DATA_REPAIR"
    QUARANTINE = "QUARANTINE"
    RERUN_UPSTREAM = "RERUN_UPSTREAM"
    ESCALATE = "ESCALATE"


class ObjectionKind(StrEnum):
    """Why the critic rejected a patch."""

    WRONG_ROOT_CAUSE = "WRONG_ROOT_CAUSE"
    UNSAFE_CODE = "UNSAFE_CODE"
    DATA_LOSS_RISK = "DATA_LOSS_RISK"
    WRONG_REMEDIATION_KIND = "WRONG_REMEDIATION_KIND"
    MISSING_TEST = "MISSING_TEST"
    OTHER = "OTHER"


class Objection(MeridianModel):
    """One typed reason the critic refused the patch."""

    kind: ObjectionKind = Field(description="Objection category.")
    message: NonBlank = Field(description="What the reviewer wants changed.")


class VerificationCheck(MeridianModel):
    """One deterministic check inside a verification."""

    name: NonBlank = Field(description="Check name, such as contracts or row_loss.")
    passed: bool = Field(description="Whether this check passed.")
    detail: str = Field(
        default="",
        description="Short explanation. Empty when there is nothing to add.",
    )
