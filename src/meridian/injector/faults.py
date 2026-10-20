"""Pure fault injectors. Same seed and inputs rebuild the same frame and label."""

from datetime import date
from uuid import uuid5

import numpy as np
import pandas as pd

from meridian.contracts.fault_type import FaultType
from meridian.contracts.injected_fault import InjectedFault
from meridian.contracts.namespace import MERIDIAN_NAMESPACE
from meridian.pipeline.catalog import require_dataset
from meridian.quality.freshness import lateness_seconds


def inject_schema_drift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    intensity: float,
    *,
    dataset: str,
    seed: int,
) -> tuple[pd.DataFrame, InjectedFault]:
    """Retype, rename, or drop ``vehicle_id`` / the first key column.

    Intensity below 0.1 retypes, below 0.5 renames, and 0.5 and above drops.
    """
    _require_rng(rng)
    _require_intensity(intensity)
    spec = require_dataset(dataset)
    column = spec.primary_key[0]
    out = df.copy()
    if intensity < 0.1:
        out[column] = np.arange(len(out), dtype=np.float64)
        columns = [column]
    elif intensity < 0.5:
        renamed = f"{column}_renamed"
        out = out.rename(columns={column: renamed})
        columns = [column, renamed]
    else:
        out = out.drop(columns=[column])
        columns = [column]
    return out, _label(
        fault=FaultType.SCHEMA_DRIFT,
        dataset=dataset,
        columns=columns,
        rows=len(df),
        intensity=intensity,
        seed=seed,
    )


def inject_null_explosion(
    df: pd.DataFrame,
    rng: np.random.Generator,
    intensity: float,
    *,
    dataset: str,
    seed: int,
) -> tuple[pd.DataFrame, InjectedFault]:
    """Set a non-nullable column to null on ``intensity`` of the rows."""
    _require_rng(rng)
    _require_intensity(intensity)
    spec = require_dataset(dataset)
    count = _row_count(len(df), intensity)
    indexes = _pick(rng, len(df), count)
    out = df.copy()
    out.loc[out.index[indexes], spec.null_column] = np.nan
    return out, _label(
        fault=FaultType.NULL_EXPLOSION,
        dataset=dataset,
        columns=[spec.null_column],
        rows=count,
        intensity=intensity,
        seed=seed,
    )


def inject_unit_mismatch(
    df: pd.DataFrame,
    rng: np.random.Generator,
    intensity: float,
    *,
    dataset: str,
    seed: int,
) -> tuple[pd.DataFrame, InjectedFault]:
    """Convert m/s to km/h, or km to m, on ``intensity`` of the rows."""
    _require_rng(rng)
    _require_intensity(intensity)
    spec = require_dataset(dataset)
    count = _row_count(len(df), intensity)
    indexes = _pick(rng, len(df), count)
    out = df.copy()
    target = out.index[indexes]
    out.loc[target, spec.unit_column] = out.loc[target, spec.unit_column] * spec.unit_factor
    return out, _label(
        fault=FaultType.UNIT_MISMATCH,
        dataset=dataset,
        columns=[spec.unit_column],
        rows=count,
        intensity=intensity,
        seed=seed,
    )


def inject_duplicate_keys(
    df: pd.DataFrame,
    rng: np.random.Generator,
    intensity: float,
    *,
    dataset: str,
    seed: int,
) -> tuple[pd.DataFrame, InjectedFault]:
    """Append copies of ``intensity`` of the rows, repeating their primary key."""
    _require_rng(rng)
    _require_intensity(intensity)
    spec = require_dataset(dataset)
    count = _row_count(len(df), intensity)
    indexes = _pick(rng, len(df), count)
    extra = df.iloc[indexes]
    out = pd.concat([df, extra], ignore_index=True)
    return out, _label(
        fault=FaultType.DUPLICATE_KEYS,
        dataset=dataset,
        columns=list(spec.primary_key),
        rows=count,
        intensity=intensity,
        seed=seed,
    )


def inject_late_partition(
    df: pd.DataFrame,
    rng: np.random.Generator,
    intensity: float,
    *,
    dataset: str,
    seed: int,
) -> tuple[pd.DataFrame, InjectedFault]:
    """Leave the frame unchanged. The label records a late arrival.

    Callers rebuild the arrival with :func:`meridian.quality.freshness.late_arrival`
    from the partition date and this intensity. The frame is copied so the
    caller's object is not aliased.
    """
    _require_rng(rng)
    _require_intensity(intensity)
    require_dataset(dataset)
    partition = date.fromisoformat(str(df["partition_date"].iloc[0]))
    delay = lateness_seconds(partition, intensity)
    return df.copy(), _label(
        fault=FaultType.LATE_PARTITION,
        dataset=dataset,
        columns=[],
        rows=0,
        intensity=intensity,
        seed=seed,
        arrival_delay_s=delay,
    )


def inject_distribution_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    intensity: float,
    *,
    dataset: str,
    seed: int,
) -> tuple[pd.DataFrame, InjectedFault]:
    """Power-warp the PSI column inside its generating range.

    The warp is ``u ** (1 + 100 * intensity)`` on the unit interval of the
    generating bounds. Values stay inside the legal contract range. On
    ``trip_summary`` the distance is recomputed from the warped speed so the
    speed/distance identity still holds. Only PSI moves past the default
    0.2 threshold.
    """
    _require_rng(rng)
    _require_intensity(intensity)
    spec = require_dataset(dataset)
    out = df.copy()
    out[spec.psi_column] = _warp(
        out[spec.psi_column].to_numpy(dtype=float),
        spec.psi_min,
        spec.psi_max,
        intensity,
    )
    columns = [spec.psi_column]
    if spec.psi_column == "avg_speed_mps":
        duration_s = (out["end_ts"] - out["start_ts"]).dt.total_seconds().to_numpy(dtype=float)
        out["distance_km"] = out[spec.psi_column].to_numpy(dtype=float) * duration_s / 1_000.0
        columns.append("distance_km")
    return out, _label(
        fault=FaultType.DISTRIBUTION_SHIFT,
        dataset=dataset,
        columns=columns,
        rows=len(df),
        intensity=intensity,
        seed=seed,
    )


INJECTORS = {
    FaultType.SCHEMA_DRIFT: inject_schema_drift,
    FaultType.NULL_EXPLOSION: inject_null_explosion,
    FaultType.UNIT_MISMATCH: inject_unit_mismatch,
    FaultType.DUPLICATE_KEYS: inject_duplicate_keys,
    FaultType.LATE_PARTITION: inject_late_partition,
    FaultType.DISTRIBUTION_SHIFT: inject_distribution_shift,
}


def _warp(values: np.ndarray, low: float, high: float, intensity: float) -> np.ndarray:
    span = high - low
    unit = np.clip((values - low) / span, 0.0, 1.0)
    exponent = 1.0 + 100.0 * intensity
    return low + np.power(unit, exponent) * span


def _row_count(rows: int, intensity: float) -> int:
    return max(1, round(intensity * rows))


def _pick(rng: np.random.Generator, rows: int, count: int) -> np.ndarray:
    return np.sort(rng.choice(rows, size=min(count, rows), replace=False))


def _require_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")


def _require_intensity(intensity: float) -> None:
    if not _is_finite(intensity) or intensity <= 0 or intensity > 1:
        raise ValueError("intensity must be in the interval (0, 1]")


def _is_finite(value: float) -> bool:
    """True for a real finite number. Rejects booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return bool(np.isfinite(value))


def _label(
    *,
    fault: FaultType,
    dataset: str,
    columns: list[str],
    rows: int,
    intensity: float,
    seed: int,
    arrival_delay_s: int | None = None,
) -> InjectedFault:
    ordered = list(columns)
    delay = "" if arrival_delay_s is None else str(arrival_delay_s)
    raw = "|".join(
        [
            dataset,
            fault.value,
            f"{intensity:.8f}",
            str(seed),
            ",".join(ordered),
            str(rows),
            delay,
        ]
    )
    return InjectedFault(
        id=uuid5(MERIDIAN_NAMESPACE, raw),
        fault_type=fault,
        dataset=dataset,
        columns_affected=ordered,
        rows_affected=rows,
        intensity=intensity,
        seed=seed,
        arrival_delay_s=arrival_delay_s,
    )
