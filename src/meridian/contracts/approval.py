"""Human approval of a patch. TIMEOUT is not an approval."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank, UtcDatetime


class ApprovalOutcome(StrEnum):
    """The decision recorded from the human approval gate."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


class ApprovalDecision(MeridianModel):
    """The human decision on whether a verified patch may merge."""

    patch_id: UUID = Field(description="Patch the human decided on.")
    decision: ApprovalOutcome = Field(description="APPROVED, REJECTED, or TIMEOUT.")
    approver: NonBlank = Field(description="Identity that decided, such as a Slack user id.")
    decided_at: UtcDatetime = Field(description="UTC time of the decision.")
    comment: str = Field(description="Human comment. Empty when none was given.")
