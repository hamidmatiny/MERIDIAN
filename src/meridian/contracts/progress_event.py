"""One recorded node boundary. The postmortem reads these and does not invent times."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import FiniteFloat, NonBlank, UtcDatetime


class ProgressEvent(MeridianModel):
    """A node start or end, written when it happens."""

    occurred_at: UtcDatetime = Field(description="UTC wall time of this boundary.")
    run_id: UUID = Field(description="Run that executed the node.")
    case_id: NonBlank = Field(description="Incident id, or '-' when the node has no single case.")
    node: NonBlank = Field(description="Graph node name.")
    attempt: int = Field(ge=1, description="Attempt number for this node.")
    phase: Literal["start", "end"] = Field(description="Whether the node is starting or finishing.")
    elapsed_s: FiniteFloat = Field(ge=0.0, description="Seconds since the process clock started.")
