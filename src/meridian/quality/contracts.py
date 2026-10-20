"""Pandera models for the three fleet datasets."""

from typing import ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import DataFrameModel, Field

from meridian.pipeline.catalog import DATASETS


class VehicleTelemetry(DataFrameModel):
    """Raw per-vehicle samples. ``speed_mps`` is meters per second, 0-60."""

    vehicle_id: str = Field(nullable=False)
    ts: pd.DatetimeTZDtype = Field(dtype_kwargs={"unit": "us", "tz": "UTC"}, nullable=False)
    speed_mps: float = Field(ge=0, le=60, nullable=False)
    accel_mps2: float = Field(ge=-12, le=12, nullable=False)
    battery_pct: float = Field(ge=0, le=100, nullable=False)
    odometer_km: float = Field(ge=0, le=500_000, nullable=False)
    lat: float = Field(ge=-90, le=90, nullable=False)
    lon: float = Field(ge=-180, le=180, nullable=False)
    partition_date: str = Field(nullable=False)

    class Config:
        """Reject undeclared columns and repeated ``(vehicle_id, ts)`` pairs."""

        strict = True
        name = "VehicleTelemetry"
        unique: ClassVar[list[str]] = ["vehicle_id", "ts"]


class TripSummary(DataFrameModel):
    """One row per trip. Average speed must match distance over duration."""

    trip_id: str = Field(nullable=False)
    vehicle_id: str = Field(nullable=False)
    start_ts: pd.DatetimeTZDtype = Field(dtype_kwargs={"unit": "us", "tz": "UTC"}, nullable=False)
    end_ts: pd.DatetimeTZDtype = Field(dtype_kwargs={"unit": "us", "tz": "UTC"}, nullable=False)
    distance_km: float = Field(ge=0, le=500, nullable=False)
    avg_speed_mps: float = Field(ge=0, le=60, nullable=False)
    partition_date: str = Field(nullable=False)

    class Config:
        """Reject undeclared columns and repeated trip ids."""

        strict = True
        name = "TripSummary"
        unique: ClassVar[list[str]] = ["trip_id"]

    @pa.dataframe_check(name="speed_matches_distance")
    @classmethod
    def speed_matches_distance(cls, df: pd.DataFrame) -> pd.Series:
        """``avg_speed_mps`` agrees with ``distance_km / duration`` within 1e-4 m/s."""
        duration_s = (df["end_ts"] - df["start_ts"]).dt.total_seconds()
        expected = df["distance_km"] * 1_000.0 / duration_s
        return (df["avg_speed_mps"] - expected).abs() <= 1e-4


class FleetDailyKpi(DataFrameModel):
    """One row per vehicle per day."""

    partition_date: str = Field(nullable=False)
    vehicle_id: str = Field(nullable=False)
    total_distance_km: float = Field(ge=0, le=2_000, nullable=False)
    trips: int = Field(ge=1, le=200, nullable=False)
    utilization_pct: float = Field(ge=0, le=100, nullable=False)

    class Config:
        """Reject undeclared columns and repeated vehicle rows in a partition."""

        strict = True
        name = "FleetDailyKpi"
        unique: ClassVar[list[str]] = ["partition_date", "vehicle_id"]


MODELS: dict[str, type[DataFrameModel]] = {
    "vehicle_telemetry": VehicleTelemetry,
    "trip_summary": TripSummary,
    "fleet_daily_kpi": FleetDailyKpi,
}


def model_for(dataset: str) -> type[DataFrameModel]:
    """Return the pandera model for ``dataset``."""
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}")
    return MODELS[dataset]
