"""A proposed fix and the regression test that guards it."""

from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.validation import NonBlank


class Patch(MeridianModel):
    """A code change proposed for one hypothesis.

    ``id`` is the value later verdicts, verification, and approval must match.
    ``diff`` adds ``remediations/<incident_id>.py``. The regression test is pandera-based.
    """

    id: UUID = Field(description="Stable identifier for this patch.")
    incident_id: UUID = Field(description="Incident this patch is meant to resolve.")
    hypothesis_id: UUID = Field(description="Hypothesis this patch implements.")
    remediation_kind: RemediationKind = Field(
        default=RemediationKind.ESCALATE,
        description=(
            "Whether this patch repairs rows, quarantines them, requests a rerun, or escalates."
        ),
    )
    diff: NonBlank = Field(description="Unified diff that adds remediations/<incident_id>.py.")
    regression_test: NonBlank = Field(description="Pandera-based test for the remediation.")
    files_touched: list[NonBlank] = Field(
        min_length=1,
        description="Repo-relative paths the diff changes.",
    )
    author_model: NonBlank = Field(description="Model that authored the patch.")
