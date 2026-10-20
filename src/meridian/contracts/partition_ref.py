"""One partition Sentinel can evaluate."""

from datetime import date

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.validation import NonBlank


class PartitionRef(MeridianModel):
    """A stored partition plus the reference profile it is scored against."""

    dataset: NonBlank = Field(description="Dataset name, matching the parquet schema.")
    partition_date: date = Field(description="Partition date the file belongs to.")
    parquet_path: NonBlank = Field(description="Path to the partition parquet file.")
    metadata: PartitionMetadata = Field(description="Arrival metadata for the freshness check.")
    reference_profile_path: NonBlank = Field(
        description="Path to a ReferenceProfile JSON fit on a different clean partition.",
    )
    generation_seed: int | None = Field(
        default=None,
        ge=0,
        description=("Pipeline seed. Absent when this partition was not generated here."),
    )
    row_count: int | None = Field(
        default=None,
        ge=1,
        description="Row count the pipeline was asked to generate.",
    )
