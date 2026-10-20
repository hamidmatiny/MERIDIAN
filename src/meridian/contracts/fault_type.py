"""Classes of data-quality failure Sentinel can emit."""

from enum import StrEnum


class FaultType(StrEnum):
    """The kind of contract failure detected on a partition."""

    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    NULL_EXPLOSION = "NULL_EXPLOSION"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    DUPLICATE_KEYS = "DUPLICATE_KEYS"
    LATE_PARTITION = "LATE_PARTITION"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
