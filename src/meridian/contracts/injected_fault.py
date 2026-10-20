"""Ground-truth label for one seeded fault injection."""

from uuid import UUID

from pydantic import Field

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
