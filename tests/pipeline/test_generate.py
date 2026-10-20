"""Lineage and generator behavior."""

from datetime import date

import pandas as pd
import pytest

from meridian.pipeline.catalog import DEFAULT_ROWS
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE, generate_dataset
from meridian.pipeline.lineage import downstream_of


def test_default_row_count_is_ten_thousand() -> None:
    assert DEFAULT_ROWS == 10_000


def test_vehicle_telemetry_blast_radius() -> None:
    assert downstream_of("vehicle_telemetry") == ["trip_summary", "fleet_daily_kpi"]


def test_downstream_is_ordered_and_terminal_datasets_are_empty() -> None:
    assert downstream_of("trip_summary") == ["fleet_daily_kpi"]
    assert downstream_of("fleet_daily_kpi") == []


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(KeyError):
        downstream_of("not_a_dataset")
    with pytest.raises(KeyError):
        generate_dataset("not_a_dataset", seed=1, rows=2)


def test_generate_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError):
        generate_dataset("vehicle_telemetry", seed=1, rows=0)
    with pytest.raises(ValueError):
        generate_dataset("vehicle_telemetry", seed=-1, rows=2)


def test_generators_are_deterministic_and_keep_column_order() -> None:
    partition = date(2026, 10, 19)
    expected = {
        "vehicle_telemetry": [
            "vehicle_id",
            "ts",
            "speed_mps",
            "accel_mps2",
            "battery_pct",
            "odometer_km",
            "lat",
            "lon",
            "partition_date",
        ],
        "trip_summary": [
            "trip_id",
            "vehicle_id",
            "start_ts",
            "end_ts",
            "distance_km",
            "avg_speed_mps",
            "partition_date",
        ],
        "fleet_daily_kpi": [
            "partition_date",
            "vehicle_id",
            "total_distance_km",
            "trips",
            "utilization_pct",
        ],
    }
    for name, columns in expected.items():
        first = generate_dataset(name, seed=3, rows=32, partition_date=partition)
        second = generate_dataset(name, seed=3, rows=32, partition_date=partition)
        pd.testing.assert_frame_equal(first, second)
        assert list(first.columns) == columns
        assert set(first["partition_date"]) == {partition.isoformat()}
    assert date(2026, 10, 19) == DEFAULT_PARTITION_DATE


def test_trip_speed_matches_distance() -> None:
    frame = generate_dataset("trip_summary", seed=4, rows=200)
    duration = (frame["end_ts"] - frame["start_ts"]).dt.total_seconds()
    expected = frame["distance_km"] * 1000.0 / duration
    assert (frame["avg_speed_mps"] - expected).abs().max() < 1e-6
