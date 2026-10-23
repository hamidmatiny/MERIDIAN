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
        description="machine_time_s + human_wait_s. Zero when those were not recorded.",
    )
    machine_time_s: int | None = Field(
        default=None,
        ge=0,
        description="Detection until slack_approval was reached. None if not recorded.",
    )
    human_wait_s: int | None = Field(
        default=None,
        ge=0,
        description="Message posted until the decision. None if not recorded.",
    )
    cost_usd: FiniteFloat = Field(ge=0.0, description="Recorded cost in US dollars.")

    @model_validator(mode="after")
    def timeline_is_ordered(self) -> Self:
        """Keep the timeline chronological, and the two waits additive."""
        times = [event.occurred_at for event in self.timeline]
        if times != sorted(times):
            raise ValueError("timeline must be ordered by occurred_at")
        machine = self.machine_time_s
        human = self.human_wait_s
        if (machine is None) != (human is None):
            raise ValueError("machine_time_s and human_wait_s are recorded together")
        if (
            machine is not None
            and human is not None
            and self.time_to_resolution_s != machine + human
        ):
            raise ValueError("time_to_resolution_s must equal machine_time_s + human_wait_s")
        return self
