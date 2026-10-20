"""Column bounds for the synthetic fleet pipeline.

Generating intervals sit inside the legal intervals. Unit injections push a
column out of its legal interval. Distribution-shift injections stay inside
the generating interval, so range checks do not see them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """How to generate and corrupt one dataset."""

    name: str
    primary_key: tuple[str, ...]
    null_column: str
    unit_column: str
    unit_factor: float
    psi_column: str
    psi_min: float
    psi_max: float


DATASETS: dict[str, DatasetSpec] = {
    "vehicle_telemetry": DatasetSpec(
        name="vehicle_telemetry",
        primary_key=("vehicle_id", "ts"),
        null_column="battery_pct",
        unit_column="speed_mps",
        unit_factor=3.6,
        psi_column="speed_mps",
        psi_min=20.0,
        psi_max=40.0,
    ),
    "trip_summary": DatasetSpec(
        name="trip_summary",
        primary_key=("trip_id",),
        null_column="vehicle_id",
        unit_column="avg_speed_mps",
        unit_factor=3.6,
        psi_column="avg_speed_mps",
        psi_min=20.0,
        psi_max=40.0,
    ),
    "fleet_daily_kpi": DatasetSpec(
        name="fleet_daily_kpi",
        primary_key=("partition_date", "vehicle_id"),
        null_column="total_distance_km",
        unit_column="total_distance_km",
        unit_factor=1000.0,
        psi_column="utilization_pct",
        psi_min=10.0,
        psi_max=90.0,
    ),
}

DEFAULT_ROWS = 10_000
PSI_BINS = 10
DEFAULT_PSI_THRESHOLD = 0.2


def require_dataset(name: str) -> DatasetSpec:
    """Return the catalog entry or raise ``KeyError``."""
    try:
        return DATASETS[name]
    except KeyError as exc:
        known = ", ".join(DATASETS)
        raise KeyError(f"unknown dataset {name!r}; expected one of {known}") from exc
