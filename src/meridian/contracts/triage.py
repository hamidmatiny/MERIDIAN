"""One structured triage decision for a single incident."""

from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from meridian.contracts.base import MeridianModel
from meridian.contracts.severity import Severity, more_severe
from meridian.contracts.validation import FiniteFloat, NonBlank


class TriageAssessment(MeridianModel):
    """The fields the triage model is asked to fill. The floor is applied afterwards."""

    incident_id: UUID = Field(description="Incident this assessment refers to.")
    severity: Severity = Field(description="Severity the model proposed.")
    rationale: NonBlank = Field(description="Why the model chose that severity.")
    affected_consumers: list[NonBlank] = Field(
        description="Downstream datasets the model believes are affected.",
    )
    confidence: FiniteFloat = Field(
        ge=0,
        le=1,
        description="Model confidence in this assessment, from 0 to 1.",
    )


class TriageResult(MeridianModel):
    """Triage output after the severity floor has been applied.

    ``severity`` and ``final_severity`` are the same value. ``llm_severity`` is
    what the model returned, and is unset when structured output never validated.
    """

    incident_id: UUID = Field(description="Incident this decision refers to.")
    severity: Severity = Field(description="Severity stored on the run. Equals final_severity.")
    rationale: NonBlank = Field(description="Why this severity was chosen.")
    affected_consumers: list[NonBlank] = Field(
        description="Downstream datasets named by triage.",
    )
    confidence: FiniteFloat = Field(ge=0, le=1, description="Confidence from 0 to 1.")
    llm_severity: Severity | None = Field(
        default=None,
        description="Severity the model returned. Unset when triage degraded.",
    )
    floor_severity: Severity = Field(description="Deterministic minimum severity.")
    final_severity: Severity = Field(
        description="The more severe of the model severity and the floor.",
    )
    prompt_version: NonBlank = Field(description="Version of the triage prompt that was loaded.")
    triage_degraded: bool = Field(
        default=False,
        description="True when structured output failed and the floor was kept.",
    )

    @model_validator(mode="after")
    def final_respects_the_floor(self) -> Self:
        """The recorded severity is the floor when degraded, otherwise the more severe value."""
        if self.severity is not self.final_severity:
            raise ValueError("severity must equal final_severity")
        if self.triage_degraded:
            if self.final_severity is not self.floor_severity:
                raise ValueError("degraded triage keeps the floor severity")
            return self
        if self.llm_severity is None:
            raise ValueError("a completed triage records llm_severity")
        expected = more_severe(self.llm_severity, self.floor_severity)
        if self.final_severity is not expected:
            raise ValueError("final_severity must be the more severe of llm_severity and the floor")
        return self
