"""The incident report written after the response loop finishes."""

from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import FiniteFloat, NonBlank, UtcDatetime


class TimelineEvent(MeridianModel):
    """One entry in an incident timeline."""

    occurred_at: UtcDatetime = Field(description="UTC time the event happened.")
    actor: NonBlank = Field(description="Agent, human, or system that caused the event.")
    summary: NonBlank = Field(description="What happened.")


class Postmortem(MeridianModel):
    """Incident report. The timeline is ordered by ``occurred_at``."""

    incident_id: UUID = Field(description="Incident this report describes.")
    timeline: list[TimelineEvent] = Field(
        default_factory=list,
        description="Events in non-decreasing occurred_at order.",
    )
    root_cause: NonBlank = Field(description="Accepted root cause.")
    fix_summary: NonBlank = Field(description="What the approved fix changed.")
    time_to_resolution_s: int = Field(
        ge=0,
        description="Whole seconds from detection to approval or escalation.",
    )
    cost_usd: FiniteFloat = Field(ge=0.0, description="Recorded cost in US dollars.")

    @model_validator(mode="after")
    def timeline_is_ordered(self) -> Self:
        """Keep the timeline chronological, allowing equal timestamps."""
        times = [event.occurred_at for event in self.timeline]
        if times != sorted(times):
            raise ValueError("timeline must be ordered by occurred_at")
        return self
