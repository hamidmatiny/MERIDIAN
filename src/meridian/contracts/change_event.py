"""One pipeline change recorded in the evidence log."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank, UtcDatetime


class EventKind(StrEnum):
    """Kind of change recorded in ``events.jsonl``."""

    SCHEMA_CHANGE = "SCHEMA_CHANGE"
    DEPLOY = "DEPLOY"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    UPSTREAM_DELAY = "UPSTREAM_DELAY"
    BACKFILL = "BACKFILL"
    DEPENDENCY_BUMP = "DEPENDENCY_BUMP"


class ChangeEvent(MeridianModel):
    """A single pipeline event. The causal event for a fault is one of these."""

    id: UUID = Field(description="Stable identifier. Tool refs use event:<id>.")
    ts: UtcDatetime = Field(description="UTC time the change was recorded.")
    kind: EventKind = Field(description="What kind of change this event is.")
    dataset: NonBlank = Field(description="Dataset the change applies to.")
    author: NonBlank = Field(description="Person or system that recorded the change.")
    summary: NonBlank = Field(description="One-line description of the change.")
    diff_excerpt: str = Field(description="Short diff or config excerpt. Empty when there is none.")
