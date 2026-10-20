"""An incident emitted when a data contract fails on a partition."""

from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.fault_type import FaultType
from meridian.contracts.severity import Severity
from meridian.contracts.validation import NonBlank, UtcDatetime

# Scalar cell values only. Nested documents are out of scope for Phase 0.
RowValue = str | int | float | bool | None

_MAX_SAMPLE_ROWS = 50


class Incident(MeridianModel):
    """A data-quality failure on one dataset partition.

    ``severity`` stays unset. The effective severity is ``IncidentCase.severity``.
    ``sample_failure_rows`` holds at most 50 scalar rows.
    """

    id: UUID = Field(description="Stable identifier for this incident.")
    detected_at: UtcDatetime = Field(description="UTC time when the failure was detected.")
    dataset: NonBlank = Field(description="Dataset the failing partition belongs to.")
    partition: NonBlank = Field(description="Partition key or path that failed.")
    contract_name: NonBlank = Field(
        description="Name of the data contract that reported the failure."
    )
    fault_type: FaultType = Field(description="Class of data-quality fault.")
    failing_checks: list[NonBlank] = Field(
        min_length=1,
        description="Names of the contract checks that failed.",
    )
    sample_failure_rows: list[dict[str, RowValue]] = Field(
        default_factory=list,
        max_length=_MAX_SAMPLE_ROWS,
        description="Up to 50 sampled rows that failed the contract. Values are scalars.",
    )
    severity: Severity | None = Field(
        default=None,
        description="Stays unset. Read severity from IncidentCase.severity.",
    )
    blast_radius: list[NonBlank] = Field(
        default_factory=list,
        description="Downstream datasets affected by this incident.",
    )
