"""Turn contract failures into Incident records."""

import math
from datetime import UTC, datetime
from uuid import UUID, uuid5

import numpy as np
import pandas as pd
from pandera.errors import SchemaError, SchemaErrorReason, SchemaErrors

from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident import Incident, RowValue
from meridian.contracts.namespace import MERIDIAN_NAMESPACE
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.pipeline.lineage import downstream_of
from meridian.quality.contracts import model_for
from meridian.quality.freshness import DEFAULT_FRESHNESS_SLA, is_late
from meridian.quality.mapping import fault_type_for
from meridian.quality.psi import population_stability_index

_MAX_SAMPLE_ROWS = 50


def evaluate(
    dataset_name: str,
    df: pd.DataFrame,
    metadata: PartitionMetadata,
    reference_profile: ReferenceProfile,
    *,
    detected_at: datetime | None = None,
) -> list[Incident]:
    """Run the dataset contract, PSI, and the freshness SLA.

    Severity stays unset. Blast radius comes from lineage. Sample rows are
    capped at 50. Each pandera failure maps through the explicit check table.
    """
    if metadata.dataset != dataset_name:
        raise ValueError("metadata.dataset must match dataset_name")
    if reference_profile.dataset != dataset_name:
        raise ValueError("reference_profile.dataset must match dataset_name")
    moment = datetime.now(UTC) if detected_at is None else detected_at
    grouped: dict[FaultType, _Bucket] = {}
    _collect_schema_errors(df, dataset_name, grouped)
    _collect_psi(df, reference_profile, grouped)
    if is_late(metadata, DEFAULT_FRESHNESS_SLA):
        _add(
            grouped,
            FaultType.LATE_PARTITION,
            "partition_arrival_sla",
            [],
        )
    partition = metadata.partition_date.isoformat()
    blast = downstream_of(dataset_name)
    incidents: list[Incident] = []
    for fault_type in sorted(grouped, key=lambda item: item.value):
        bucket = grouped[fault_type]
        incidents.append(
            Incident(
                id=_incident_id(
                    dataset_name,
                    partition,
                    model_for(dataset_name).__name__,
                    fault_type,
                ),
                detected_at=moment,
                dataset=dataset_name,
                partition=partition,
                contract_name=model_for(dataset_name).__name__,
                fault_type=fault_type,
                failing_checks=sorted(bucket.checks),
                sample_failure_rows=_rows_from(df, bucket.indices),
                severity=None,
                blast_radius=blast,
            )
        )
    return incidents


def failure_positions(
    dataset_name: str,
    df: pd.DataFrame,
    metadata: PartitionMetadata,
    reference_profile: ReferenceProfile,
    fault: FaultType,
) -> set[int]:
    """Row positions that fail ``fault`` on this frame. The injector is not consulted."""
    if metadata.dataset != dataset_name:
        raise ValueError("metadata.dataset must match dataset_name")
    if reference_profile.dataset != dataset_name:
        raise ValueError("reference_profile.dataset must match dataset_name")
    grouped: dict[FaultType, _Bucket] = {}
    _collect_schema_errors(df, dataset_name, grouped)
    _collect_psi(df, reference_profile, grouped)
    if is_late(metadata, DEFAULT_FRESHNESS_SLA):
        _add(grouped, FaultType.LATE_PARTITION, "partition_arrival_sla", [])
    bucket = grouped.get(fault)
    if bucket is None:
        return set()
    return set(bucket.indices)


def _incident_id(dataset: str, partition: str, contract_name: str, fault: FaultType) -> UUID:
    """UUID5 of the partition and the fault. The same inputs rebuild the same id."""
    payload = "|".join((dataset, partition, contract_name, fault.value))
    return uuid5(MERIDIAN_NAMESPACE, payload)


class _Bucket:
    def __init__(self) -> None:
        self.checks: set[str] = set()
        self.indices: list[int] = []


def _add(
    grouped: dict[FaultType, _Bucket],
    fault: FaultType,
    check: str,
    indices: list[int],
) -> None:
    bucket = grouped.setdefault(fault, _Bucket())
    bucket.checks.add(check)
    for index in indices:
        if index not in bucket.indices:
            bucket.indices.append(index)


def _collect_schema_errors(
    df: pd.DataFrame,
    dataset_name: str,
    grouped: dict[FaultType, _Bucket],
) -> None:
    try:
        model_for(dataset_name).validate(df, lazy=True)
    except SchemaErrors as exc:
        for error in exc.schema_errors:
            name = _check_name(error)
            reason = error.reason_code if isinstance(error.reason_code, SchemaErrorReason) else None
            fault = fault_type_for(name, reason)
            label = name if name is not None else (reason.value if reason is not None else "schema")
            column = error.column_name if isinstance(error.column_name, str) else None
            if column:
                label = f"{column}.{label}"
            _add(grouped, fault, label, _indices(error))


def _collect_psi(
    df: pd.DataFrame,
    profile: ReferenceProfile,
    grouped: dict[FaultType, _Bucket],
) -> None:
    for column in profile.columns:
        if column.column not in df.columns:
            continue
        score = population_stability_index(df[column.column].to_numpy(dtype=float), column)
        if score >= profile.psi_threshold:
            check = f"psi_{column.column}"
            fault = fault_type_for(check, None)
            _add(grouped, fault, check, list(range(min(_MAX_SAMPLE_ROWS, len(df)))))


def _check_name(error: SchemaError) -> str | None:
    check = error.check
    if isinstance(check, str):
        return check.split("(", maxsplit=1)[0]
    name = getattr(check, "name", None)
    if isinstance(name, str):
        return name
    return None


def _indices(error: SchemaError) -> list[int]:
    cases = error.failure_cases
    if not isinstance(cases, pd.DataFrame) or "index" not in cases.columns:
        return []
    found: list[int] = []
    for value in cases["index"].tolist():
        if _missing(value):
            continue
        found.append(int(value))
    return found


def _missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def _rows_from(df: pd.DataFrame, indices: list[int]) -> list[dict[str, RowValue]]:
    if not indices:
        return []
    chosen = [index for index in indices if index in df.index][:_MAX_SAMPLE_ROWS]
    if not chosen:
        return []
    records = df.loc[chosen].to_dict(orient="records")
    return [{str(key): _scalar(value) for key, value in row.items()} for row in records]


def _scalar(value: object) -> RowValue:
    if _missing(value):
        return None
    if isinstance(value, bool | int | float | str):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _scalar(value.item())
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        text = isoformat()
        if isinstance(text, str):
            return text
    return str(value)
