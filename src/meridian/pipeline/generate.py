"""Deterministic generators for the three fleet datasets."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from meridian.pipeline.catalog import DEFAULT_ROWS, require_dataset

DEFAULT_PARTITION_DATE = date(2026, 10, 19)


def generate_dataset(
    name: str,
    seed: int,
    rows: int = DEFAULT_ROWS,
    partition_date: date = DEFAULT_PARTITION_DATE,
) -> pd.DataFrame:
    """Build ``rows`` of ``name`` from ``numpy.random.Generator(seed)``.

    Trip and KPI frames are generated at their own grain so a caller can ask
    for an exact row count. Their values still satisfy the physical checks
    (speed matches distance over duration; ranges stay inside the contract).
    """
    if rows < 1:
        raise ValueError("rows must be at least 1")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    spec = require_dataset(name)
    rng = np.random.default_rng(seed)
    if spec.name == "vehicle_telemetry":
        frame = _vehicle_telemetry(rng, rows, partition_date)
    elif spec.name == "trip_summary":
        frame = _trip_summary(rng, rows, partition_date)
    else:
        frame = _fleet_daily_kpi(rng, rows, partition_date)
    frame.attrs["dataset"] = spec.name
    return frame


def _midnight(partition_date: date) -> datetime:
    return datetime(partition_date.year, partition_date.month, partition_date.day, tzinfo=UTC)


def _vehicle_telemetry(rng: np.random.Generator, rows: int, partition_date: date) -> pd.DataFrame:
    start = _midnight(partition_date)
    timestamps = pd.to_datetime(
        [start + timedelta(seconds=index) for index in range(rows)],
        utc=True,
    )
    return pd.DataFrame(
        {
            "vehicle_id": [f"veh-{index % 100:04d}" for index in range(rows)],
            "ts": timestamps,
            "speed_mps": rng.uniform(20.0, 40.0, rows),
            "accel_mps2": rng.uniform(-4.0, 4.0, rows),
            "battery_pct": rng.uniform(10.0, 90.0, rows),
            "odometer_km": rng.uniform(0.0, 100_000.0, rows),
            "lat": rng.uniform(25.0, 48.0, rows),
            "lon": rng.uniform(-124.0, -70.0, rows),
            "partition_date": [partition_date.isoformat()] * rows,
        }
    )


def _trip_summary(rng: np.random.Generator, rows: int, partition_date: date) -> pd.DataFrame:
    start = _midnight(partition_date)
    duration_s = rng.uniform(120.0, 1_800.0, rows)
    avg_speed = rng.uniform(20.0, 40.0, rows)
    distance_km = avg_speed * duration_s / 1_000.0
    offset_s = np.arange(rows) % 70_000
    start_ts = pd.to_datetime(
        [start + timedelta(seconds=int(offset)) for offset in offset_s],
        utc=True,
    )
    end_ts = (start_ts + pd.to_timedelta(duration_s, unit="s")).astype("datetime64[us, UTC]")
    return pd.DataFrame(
        {
            "trip_id": [f"trip-{index:06d}" for index in range(rows)],
            "vehicle_id": [f"veh-{index % 100:04d}" for index in range(rows)],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "distance_km": distance_km,
            "avg_speed_mps": avg_speed,
            "partition_date": [partition_date.isoformat()] * rows,
        }
    )


def _fleet_daily_kpi(rng: np.random.Generator, rows: int, partition_date: date) -> pd.DataFrame:
    partition = partition_date.isoformat()
    return pd.DataFrame(
        {
            "partition_date": [partition] * rows,
            "vehicle_id": [f"veh-{index:05d}" for index in range(rows)],
            "total_distance_km": rng.uniform(50.0, 400.0, rows),
            "trips": rng.integers(1, 31, rows),
            "utilization_pct": rng.uniform(10.0, 90.0, rows),
        }
    )
