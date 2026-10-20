"""Arrival metadata for one dataset partition."""

from datetime import date

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank, UtcDatetime


class PartitionMetadata(MeridianModel):
    """When a partition was produced and when it arrived."""

    dataset: NonBlank = Field(description="Dataset this partition belongs to.")
    partition_date: date = Field(description="Calendar date of the partition.")
    arrived_at: UtcDatetime = Field(description="UTC time the partition arrived.")
