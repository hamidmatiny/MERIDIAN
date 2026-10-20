"""Ground-truth label for one seeded fault injection."""

from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from meridian.contracts.base import MeridianModel
from meridian.contracts.fault_type import FaultType
from meridian.contracts.validation import FiniteFloat, NonBlank


class InjectedFault(MeridianModel):
    """The label MERIDIAN-Bench will score detections against.

    The id is a UUID5 of the injection inputs, so the same inputs rebuild the
    same label.
    """

    id: UUID = Field(description="Deterministic identifier for this injection.")
    fault_type: FaultType = Field(description="Fault that was applied.")
    dataset: NonBlank = Field(description="Dataset the fault was applied to.")
    columns_affected: list[NonBlank] = Field(
        description="Columns the injector changed. Empty when only metadata moved.",
    )
    rows_affected: int = Field(ge=0, description="How many rows the injector changed.")
    intensity: FiniteFloat = Field(
        gt=0,
        le=1,
        description="Strength of the injection, from just above 0 through 1.",
    )
    seed: int = Field(ge=0, description="Seed used to build the corrupted frame.")
    arrival_delay_s: int | None = Field(
        default=None,
        ge=0,
        description="Seconds past the freshness deadline. Set only for a late partition.",
    )
    causal_event_id: UUID = Field(
        description="The one change event that caused this fault. It must be in events.jsonl.",
    )

    @model_validator(mode="after")
    def late_partition_records_its_delay(self) -> Self:
        """A late partition must record how late it was. Other faults must not."""
        if self.fault_type is FaultType.LATE_PARTITION and self.arrival_delay_s is None:
            raise ValueError("LATE_PARTITION requires arrival_delay_s")
        if self.fault_type is not FaultType.LATE_PARTITION and self.arrival_delay_s is not None:
            raise ValueError("arrival_delay_s is only allowed for LATE_PARTITION")
        return self
