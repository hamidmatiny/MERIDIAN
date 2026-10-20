"""Every injected fault is caught as that fault type."""

from datetime import UTC, datetime

import numpy as np
import pytest

from meridian.contracts.fault_type import FaultType
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.injector.faults import INJECTORS
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE, generate_dataset
from meridian.pipeline.lineage import downstream_of
from meridian.quality.contracts import model_for
from meridian.quality.evaluate import evaluate
from meridian.quality.freshness import late_arrival, on_time_arrival
from meridian.quality.reference import fit_reference_profile

DATASETS = ("vehicle_telemetry", "trip_summary", "fleet_daily_kpi")
INTENSITIES = (0.01, 0.1, 0.5)
ROWS = 2_000
SEED = 42
HOLDOUT_SEED = 9_001
DETECTED_AT = datetime(2026, 10, 20, 3, 0, tzinfo=UTC)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("fault", list(FaultType))
@pytest.mark.parametrize("intensity", INTENSITIES)
def test_injected_fault_is_detected(dataset: str, fault: FaultType, intensity: float) -> None:
    frame = generate_dataset(dataset, SEED, rows=ROWS)
    corrupted, label = INJECTORS[fault](
        frame,
        np.random.default_rng(SEED),
        intensity,
        dataset=dataset,
        seed=SEED,
    )
    arrived = (
        late_arrival(DEFAULT_PARTITION_DATE, intensity)
        if fault is FaultType.LATE_PARTITION
        else on_time_arrival(DEFAULT_PARTITION_DATE)
    )
    incidents = evaluate(
        dataset,
        corrupted,
        PartitionMetadata(
            dataset=dataset,
            partition_date=DEFAULT_PARTITION_DATE,
            arrived_at=arrived,
        ),
        fit_reference_profile(generate_dataset(dataset, HOLDOUT_SEED, rows=ROWS)),
        detected_at=DETECTED_AT,
    )
    assert label.fault_type is fault
    assert incidents
    assert all(item.fault_type is fault for item in incidents)
    assert all(item.severity is None for item in incidents)
    assert all(item.blast_radius == downstream_of(dataset) for item in incidents)
    assert all(len(item.sample_failure_rows) <= 50 for item in incidents)
    if fault is FaultType.DISTRIBUTION_SHIFT:
        model_for(dataset).validate(corrupted)
        assert {item.fault_type for item in incidents} == {FaultType.DISTRIBUTION_SHIFT}


def test_the_same_fault_keeps_its_incident_id() -> None:
    dataset = "vehicle_telemetry"
    frame = generate_dataset(dataset, SEED, rows=ROWS)
    corrupted, _label = INJECTORS[FaultType.NULL_EXPLOSION](
        frame,
        np.random.default_rng(SEED),
        0.1,
        dataset=dataset,
        seed=SEED,
    )
    profile = fit_reference_profile(generate_dataset(dataset, HOLDOUT_SEED, rows=ROWS))
    metadata = PartitionMetadata(
        dataset=dataset,
        partition_date=DEFAULT_PARTITION_DATE,
        arrived_at=on_time_arrival(DEFAULT_PARTITION_DATE),
    )
    first = evaluate(dataset, corrupted, metadata, profile, detected_at=DETECTED_AT)
    second = evaluate(dataset, corrupted, metadata, profile, detected_at=DETECTED_AT)
    assert first and second
    assert first[0].fault_type is FaultType.NULL_EXPLOSION
    assert first[0].id == second[0].id
