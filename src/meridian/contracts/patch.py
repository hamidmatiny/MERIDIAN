"""A proposed fix and the regression test that guards it."""

from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank


class Patch(MeridianModel):
    """A code change proposed for one hypothesis.

    ``id`` is the value later verdicts, verification, and approval must match.
    """

    id: UUID = Field(description="Stable identifier for this patch.")
    incident_id: UUID = Field(description="Incident this patch is meant to resolve.")
    hypothesis_id: UUID = Field(description="Hypothesis this patch implements.")
    diff: NonBlank = Field(description="Unified diff of the proposed change.")
    regression_test: NonBlank = Field(description="Source of the generated regression test.")
    files_touched: list[NonBlank] = Field(
        min_length=1,
        description="Repo-relative paths the diff changes.",
    )
    author_model: NonBlank = Field(description="Model that authored the patch.")
