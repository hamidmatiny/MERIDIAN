"""Cross-provider review of a patch."""

from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.remediation import Objection
from meridian.contracts.validation import NonBlank


class Verdict(MeridianModel):
    """An independent reviewer's decision on one patch.

    ``round`` is the Fix/Critic round and must be from 1 through 3.
    """

    patch_id: UUID = Field(description="Patch under review.")
    reviewer_model: NonBlank = Field(description="Model that produced this verdict.")
    approved: bool = Field(description="Whether the reviewer accepts the patch.")
    objections: list[Objection] = Field(
        default_factory=list,
        description="Typed reasons the reviewer rejected the patch, if any.",
    )
    round: int = Field(ge=1, le=3, description="Critic round, from 1 through 3.")
