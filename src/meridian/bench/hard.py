"""Clean partitions with realistic noise. No fault is injected.

A Sentinel incident on one of these partitions is a false positive.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.partition_ref import PartitionRef
from meridian.injector.events import build_events, write_events
from meridian.pipeline.generate import generate_dataset
from meridian.pipeline.parquet import write_parquet
from meridian.quality.freshness import deadline
from meridian.quality.reference import fit_reference_profile
from meridian.run import PreparedPartition

# Inclusive legal ranges from the pandera contracts.
_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "vehicle_telemetry": {
        "speed_mps": (0.0, 60.0),
        "accel_mps2": (-12.0, 12.0),
        "battery_pct": (0.0, 100.0),
        "odometer_km": (0.0, 500_000.0),
        "lat": (-90.0, 90.0),
        "lon": (-180.0, 180.0),
    },
    "trip_summary": {"avg_speed_mps": (0.0, 60.0)},
    "fleet_daily_kpi": {
        "total_distance_km": (0.0, 2_000.0),
        "utilization_pct": (0.0, 100.0),
    },
}


def within_sla_late(partition_date: date) -> datetime:
    """One minute before the SLA deadline. Late relative to the usual arrival, still on time."""
    return deadline(partition_date) - timedelta(seconds=60)


def hard_frame(dataset: str, seed: int, rows: int, partition_date: date) -> pd.DataFrame:
    """Gaussian noise, clipped heavy tails, exact bounds, and sparse non-null categories."""
    frame = generate_dataset(dataset, seed, rows=rows, partition_date=partition_date).copy()
    rng = np.random.default_rng(seed + 17)
    for column, (low, high) in _BOUNDS[dataset].items():
        frame[column] = _perturb(frame[column].to_numpy(dtype=float), low, high, rng)
    if dataset == "trip_summary":
        _retie_trip_speed(frame)
    if dataset != "fleet_daily_kpi":
        frame["vehicle_id"] = _sparse_ids(rng, rows)
    frame.attrs["dataset"] = dataset
    return frame


def prepare_hard(
    root: Path,
    *,
    dataset: str,
    seed: int,
    rows: int,
    partition: date,
) -> PreparedPartition:
    """Write a noisy clean partition. ``label`` stays None: nothing was injected."""
    folder = root
    frame = hard_frame(dataset, seed, rows, partition)
    events, _causal = build_events(dataset=dataset, seed=seed, fault=None, columns=[])
    write_events(folder / "events.jsonl", events)
    holdout = generate_dataset(dataset, seed + 1, rows=rows, partition_date=partition)
    profile = fit_reference_profile(holdout)
    parquet_path = folder / f"{dataset}.parquet"
    profile_path = folder / "reference_profile.json"
    write_parquet(frame, parquet_path)
    profile_path.write_text(profile.model_dump_json(), encoding="utf-8")
    return PreparedPartition(
        ref=PartitionRef(
            dataset=dataset,
            partition_date=partition,
            parquet_path=str(parquet_path),
            metadata=PartitionMetadata(
                dataset=dataset,
                partition_date=partition,
                arrived_at=within_sla_late(partition),
            ),
            reference_profile_path=str(profile_path),
            generation_seed=seed,
            row_count=rows,
        ),
        clean=frame,
        label=None,
    )


def _perturb(values: np.ndarray, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    width = high - low
    noisy = values + rng.normal(0.0, 0.02 * width, size=len(values))
    tail = rng.random(len(values)) < 0.01
    if int(tail.sum()) > 0:
        noisy[tail] = values[tail] + rng.laplace(0.0, 0.15 * width, size=int(tail.sum()))
    clipped = np.clip(noisy, low, high)
    if len(clipped) >= 2:
        clipped[0] = low
        clipped[-1] = high
    return clipped


def _retie_trip_speed(frame: pd.DataFrame) -> None:
    """Keep ``avg_speed_mps`` equal to distance over duration after the noise."""
    duration_s = (frame["end_ts"] - frame["start_ts"]).dt.total_seconds().to_numpy(dtype=float)
    duration_s = np.maximum(duration_s, 1.0)
    speed = np.clip(frame["avg_speed_mps"].to_numpy(dtype=float), 0.0, 60.0)
    distance = speed * duration_s / 1_000.0
    over = distance > 500.0
    if int(over.sum()) > 0:
        distance[over] = 500.0
        speed[over] = distance[over] * 1_000.0 / duration_s[over]
    frame["avg_speed_mps"] = speed
    frame["distance_km"] = distance


def _sparse_ids(rng: np.random.Generator, rows: int) -> list[str]:
    """Zipf ids: a few categories are common and many appear once. None are null."""
    drawn = rng.zipf(2.0, size=rows)
    return [f"veh-{int(value):04d}" for value in drawn]
