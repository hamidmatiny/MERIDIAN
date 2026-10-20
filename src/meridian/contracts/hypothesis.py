"""A root-cause hypothesis and the evidence behind it."""

from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.grounding import GroundingViolation
from meridian.contracts.validation import FiniteFloat, NonBlank


class Evidence(MeridianModel):
    """One citation supporting a root-cause hypothesis."""

    source: NonBlank = Field(description="System the evidence came from, such as logs or git.")
    reference: NonBlank = Field(
        description="Locator for the evidence, such as a path, URL, or commit.",
    )
    excerpt: str = Field(description="Short quoted excerpt. Empty when the reference stands alone.")


class Hypothesis(MeridianModel):
    """A proposed root cause for one incident.

    ``id`` is the value ``Patch.hypothesis_id`` must match.
    """

    id: UUID = Field(description="Stable identifier. Patches reference this value.")
    incident_id: UUID = Field(description="Incident this hypothesis explains.")
    root_cause_summary: NonBlank = Field(description="Plain-language root cause.")
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence items that support the summary.",
    )
    confidence: FiniteFloat = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this hypothesis, from 0.0 through 1.0.",
    )
    causal_event_id: UUID | None = Field(
        default=None,
        description="Change event the hypothesis names as the cause, if any.",
    )
    suspect_columns: list[NonBlank] = Field(
        default_factory=list,
        description="Columns the hypothesis blames.",
    )
    mechanism: NonBlank = Field(
        default="unspecified",
        description="Short description of how the cause produced the fault.",
    )
    grounding_violations: int = Field(
        default=0,
        ge=0,
        description="Evidence items or a causal id dropped because no tool returned that ref.",
    )
    grounding_findings: list[GroundingViolation] = Field(
        default_factory=list,
        description="Each dropped citation, including the raw model value and a ref sample.",
    )
