"""Clean data must not raise an incident."""

from datetime import UTC, datetime

import pytest

from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE, generate_dataset
from meridian.quality.evaluate import evaluate
from meridian.quality.freshness import on_time_arrival
from meridian.quality.reference import reference_profile

DATASETS = ("vehicle_telemetry", "trip_summary", "fleet_daily_kpi")
CLEAN_ROWS = 2_000
DETECTED_AT = datetime(2026, 10, 20, 3, 0, tzinfo=UTC)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("seed", range(20))
def test_clean_partitions_produce_zero_incidents(dataset: str, seed: int) -> None:
    frame = generate_dataset(dataset, seed, rows=CLEAN_ROWS)
    metadata = PartitionMetadata(
        dataset=dataset,
        partition_date=DEFAULT_PARTITION_DATE,
        arrived_at=on_time_arrival(DEFAULT_PARTITION_DATE),
    )
    incidents = evaluate(
        dataset,
        frame,
        metadata,
        reference_profile(dataset),
        detected_at=DETECTED_AT,
    )
    assert incidents == []
