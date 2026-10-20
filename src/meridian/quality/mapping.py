"""Explicit map from a pandera check or detector name to one FaultType.

Precedence, applied in order:

1. ``check_error`` is schema drift. The check could not run, usually because
   the column's dtype changed.
2. Otherwise the check name in ``CHECK_TO_FAULT``.
3. Otherwise the pandera reason code in ``REASON_TO_FAULT``.

There is no fallback guess. An unknown pair raises ``KeyError``.
"""

from pandera.errors import SchemaErrorReason

from meridian.contracts.fault_type import FaultType

CHECK_TO_FAULT: dict[str, FaultType] = {
    "not_nullable": FaultType.NULL_EXPLOSION,
    "multiple_fields_uniqueness": FaultType.DUPLICATE_KEYS,
    "column_in_schema": FaultType.SCHEMA_DRIFT,
    "column_in_dataframe": FaultType.SCHEMA_DRIFT,
    "greater_than_or_equal_to": FaultType.UNIT_MISMATCH,
    "less_than_or_equal_to": FaultType.UNIT_MISMATCH,
    "speed_matches_distance": FaultType.UNIT_MISMATCH,
    "psi_speed_mps": FaultType.DISTRIBUTION_SHIFT,
    "psi_avg_speed_mps": FaultType.DISTRIBUTION_SHIFT,
    "psi_utilization_pct": FaultType.DISTRIBUTION_SHIFT,
    "partition_arrival_sla": FaultType.LATE_PARTITION,
}

REASON_TO_FAULT: dict[SchemaErrorReason, FaultType] = {
    SchemaErrorReason.CHECK_ERROR: FaultType.SCHEMA_DRIFT,
    SchemaErrorReason.WRONG_DATATYPE: FaultType.SCHEMA_DRIFT,
    SchemaErrorReason.COLUMN_NOT_IN_SCHEMA: FaultType.SCHEMA_DRIFT,
    SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME: FaultType.SCHEMA_DRIFT,
    SchemaErrorReason.SERIES_CONTAINS_NULLS: FaultType.NULL_EXPLOSION,
    SchemaErrorReason.DUPLICATES: FaultType.DUPLICATE_KEYS,
}


def fault_type_for(check_name: str | None, reason: SchemaErrorReason | None) -> FaultType:
    """Look up one failure. Raise if the table does not name it."""
    if reason is SchemaErrorReason.CHECK_ERROR:
        return REASON_TO_FAULT[SchemaErrorReason.CHECK_ERROR]
    if check_name is not None and check_name in CHECK_TO_FAULT:
        return CHECK_TO_FAULT[check_name]
    if reason is not None and reason in REASON_TO_FAULT:
        return REASON_TO_FAULT[reason]
    raise KeyError(f"no FaultType mapping for check {check_name!r} reason {reason!r}")
